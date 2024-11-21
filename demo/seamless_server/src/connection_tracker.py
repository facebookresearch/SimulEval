from logging import Logger
import time


class StreamingConnectionInfo:
    def __init__(self, address, active_connections, latest_message_received_timestamp):
        self.address = address
        self.active_connections = active_connections
        self.latest_message_received_timestamp = latest_message_received_timestamp

    def __repr__(self):
        return str(self)

    def __str__(self):
        return str(
            {
                "address": self.address,
                "active_connections": self.active_connections,
                "latest_message_received_timestamp": self.latest_message_received_timestamp,
            }
        )


class ConnectionTracker:
    def __init__(self, logger: Logger):
        self.connections = dict()
        self.logger = logger

    def __str__(self):
        return str(self.connections)

    def add_connection(self, address):
        if address not in self.connections:
            self.connections[address] = StreamingConnectionInfo(address, 1, time.time())
        else:
            self.connections[address].active_connections += 1
            self.connections[address].latest_message_received_timestamp = time.time()

    def log_recent_message(self, address):
        if address in self.connections:
            self.connections[address].latest_message_received_timestamp = time.time()
        else:
            self.logger.warning(
                f"Address {address} not found in connection tracker when attempting to log recent message"
            )

    def remove_connection(self, address):
        if address in self.connections:
            self.connections[address].active_connections -= 1
            if self.connections[address].active_connections < 0:
                self.logger.warning(
                    f"Address {address} has negative active connections ({self.connections[address].active_connections})"
                )
            if self.connections[address].active_connections <= 0:
                del self.connections[address]
        else:
            self.logger.warning(
                f"Address {address} not found in connection tracker when attempting to remove it"
            )

    def get_active_connection_count(self):
        return sum(
            [connection.active_connections for connection in self.connections.values()]
        )
