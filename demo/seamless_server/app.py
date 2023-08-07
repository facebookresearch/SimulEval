from logging.config import dictConfig
from flask import Flask
from flask_sockets import Sockets
from SimulevalAgentDirectory import (
    SimulevalAgentDirectory,
    NoAvailableAgentException,
)


from SimulevalAgentDirectory import SimulevalAgentDirectory
from src.connection_tracker import ConnectionTracker

from src.simuleval_transcoder import SimulevalTranscoder
import json
import logging
from werkzeug.routing import Rule
import time

dictConfig(
    {
        "version": 1,
        "formatters": {
            "default": {
                "format": "[%(asctime)s] %(levelname)s in %(module)s: %(message)s",
            }
        },
        "handlers": {
            "wsgi": {
                "class": "logging.StreamHandler",
                "stream": "ext://flask.logging.wsgi_errors_stream",
                "formatter": "default",
            }
        },
        "root": {"level": "INFO", "handlers": ["wsgi"]},
    }
)


app = Flask(__name__)
sockets = Sockets(app)
app.logger.setLevel(logging.INFO)

available_agents = SimulevalAgentDirectory()

connection_tracker = ConnectionTracker(app.logger)


def start_seamless_stream_s2t(ws):
    app.logger.info("WS Connection accepted")
    remote_address = ws.environ.get("REMOTE_ADDR")

    connection_tracker.add_connection(remote_address)

    app.logger.info("Current connection tracker info:")
    app.logger.info(str(connection_tracker))

    transcoder = None
    debug = False
    async_processing = False

    def log_debug(*args):
        if debug:
            app.logger.info(*args)

    def ws_send(obj):
        to_send = json.dumps(obj)
        log_to_send = to_send
        if "sample_rate" in to_send:
            # don't log the speech payload
            log_to_send = json.dumps(
                {k: v for k, v in obj.items() if k is not "payload"}
            )
        log_debug(f"Gonna send to client: {log_to_send}")
        ws.send(to_send)

    latency_sent = False

    while not ws.closed:
        message = ws.receive()
        if message is None:
            log_debug("No message received...")
            continue

        connection_tracker.log_recent_message(remote_address)

        if transcoder:
            speech_and_text_output = transcoder.get_buffered_output()
            if speech_and_text_output is not None:
                lat = None
                if speech_and_text_output.speech_samples:
                    to_send = {
                        "event": "translation_speech",
                        "payload": speech_and_text_output.speech_samples,
                        "sample_rate": speech_and_text_output.speech_sample_rate,
                    }
                elif speech_and_text_output.text:
                    to_send = {
                        "event": "translation_text",
                        "payload": speech_and_text_output.text,
                    }
                else:
                    app.logger.warn(
                        "Got model output with neither speech nor text content"
                    )
                    to_send = {}  # unexpected case, but not breaking the flow
                to_send["eos"] = speech_and_text_output.final

                to_send[
                    "server_active_connections"
                ] = connection_tracker.get_active_connection_count()

                if not latency_sent:
                    lat = transcoder.first_translation_time()
                    latency_sent = True
                    to_send["latency"] = lat

                ws_send(to_send)

        if isinstance(message, bytearray) and transcoder is not None:
            transcoder.process_incoming_bytes(message)
        else:
            data = json.loads(message)
            if data["event"] == "config":
                app.logger.debug("Received ws config")
                debug = data.get("debug")
                async_processing = data.get("async_processing")

                source_language_2_letter = data.get("source_language")[:2]
                target_language_2_letter = data.get("target_language")[:2]

                # Currently s2s or s2t
                model_type = data.get("model_type")

                try:
                    agent = available_agents.get_agent_or_throw(
                        model_type,
                        source_language_2_letter,
                        target_language_2_letter,
                    )
                except NoAvailableAgentException as e:
                    app.logger.warn(f"Error while getting agent: {e}")
                    ws_send({"event": "error", "payload": str(e)})
                    ws.close()
                    break

                t0 = time.time()
                transcoder = SimulevalTranscoder(
                    agent,
                    data["rate"],
                    debug=debug,
                    buffer_limit=int(data["buffer_limit"]),
                )
                t1 = time.time()
                log_debug(f"Booting up VAD and transcoder took {t1-t0} sec")
                ws_send({"event": "server_ready"})
                if async_processing:
                    transcoder.start()

            if data["event"] == "closed":
                transcoder.close = True
                log_debug("Closed Message received: {}".format(message))
                ws.close()
                break

        if transcoder and not async_processing:
            transcoder.process_pipeline_once()

        if transcoder and transcoder.close:
            ws.close()

    app.logger.info("WS Connection closed")

    connection_tracker.remove_connection(remote_address)
    app.logger.info("Current connection tracker info:")
    app.logger.info(str(connection_tracker))

    if transcoder:
        log_debug("closing transcoder")
        transcoder.close = True


sockets.url_map.add(
    Rule(
        "/api/seamless_stream_es_en_s2t",
        endpoint=start_seamless_stream_s2t,
        websocket=True,
    )
)

if __name__ == "__main__":
    # Build all the agents before starting the server
    # s2t:
    available_agents.add_agent(
        SimulevalTranscoder.build_agent(SimulevalAgentDirectory.s2t_es_en_emma_agent),
        SimulevalAgentDirectory.s2t_es_en_emma_agent,
        "s2t",
        "es",
        "en",
    )
    available_agents.add_agent(
        SimulevalTranscoder.build_agent(SimulevalAgentDirectory.s2t_en_es_emma_agent),
        SimulevalAgentDirectory.s2t_en_es_emma_agent,
        "s2t",
        "en",
        "es",
    )
    # s2s:
    available_agents.add_agent(
        SimulevalTranscoder.build_agent(SimulevalAgentDirectory.s2s_es_en_emma_agent),
        SimulevalAgentDirectory.s2s_es_en_emma_agent,
        "s2s",
        "es",
        "en",
    )

    from gevent import pywsgi
    from geventwebsocket.handler import WebSocketHandler

    server = pywsgi.WSGIServer(("0.0.0.0", 8000), app, handler_class=WebSocketHandler)
    app.logger.info("Starting server on port 8000...")
    server.serve_forever()
