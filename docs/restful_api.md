# Introduction to the evaluation interface

Assuming the server address is *server_url*

---

## Get number of sentences in test set

* **Method:** : `GET`

* **Endpoint** : *server_url*

* **URL Params:**

* **Success Response:**

    ```
    {
        "num_sentences": N
    }
    ```


Where `N` is the number of sentences in test set

---
## Start a new evaluation session
* **Method:** : `POST`

*  **Endpoint** : *server_url*

*  **URL Params:**

Should be used every time starting to evaluate a new model.

___
## Obtain a source word from server

* **Method:** : `GET`

*  **Endpoint** : *server_url*/src

*  **URL Params:**

    ```
    {
        "instance_id": i
    }
    ```

* **Success Response:**

    ```
    {
        "instance_id": i,
        "segment_id": segment_id,
        "segment": word
    }
    ```

Using when the source data type is text. A source word in sentence *i* will be received from server.
Notice that the word is not tokenized.
```segment_id``` is the position of the word in sentence, starting from zero.
The server will automatically move the pointer to next word.
When the pointer reach the end of sentence, "<\/s>" will be sent.
For example, for sentence "A B C".
The first time the request happens, "A" will be received,  the second time will be "B", the third time will be "C",  and after that, it will be always "<\/s>". The request should be made every time the model decides to read a new source word.

---
## Obtain a segment of speech from server

* **Method:** : `GET`

*  **Endpoint** : *server_url*/src

*  **URL Params:**

    ```
    {
        "instance_id": i,
        "segment_size": t_ms
    }
    ```

* **Success Response:**

    ```
    {
        "instance_id": i,
        "segment_id": segment_id,
        "segment": list_of_numbers,
        "sample_rate": sample_rate,
        "dtype": data_type,
        "finished": is_finished,
    }
    ```

Using when the source data type is speech. The wav form for the segment of a speech utterance will be received in the format of list of numbers.
The sample rate on the server is 16000Hz. By default, the length of list is 160, or 10 ms in time.
There is an optional `segment_size` parameter in the request, the unit is `ms`. A customized length of segment can be requested. However, the length of a segment in time can only be a multiple of `10ms`. It will be rounded if not. For example, if `t_ms = 301` is requested, the returned segment will be `300ms` long.

Again, the request should be made every time the model decides to read a new segment of speech utterance.

---

## Send a translated token to the server
* **Method:** : `PUT`

*  **Endpoint** : *server_url*/hypo

*  **URL Params:**

    ```
    {
        "instance_id": i,
    }
    ```
*  **Body** (raw text)

    ```
    WORD
    ```

After the word is sent, the server will record the delay (length of source context the model used) to predict the token. Notice that the `WORD` should be detokenized if the `--eval-latency-unit` is set to `word`. If there is a space in `WORD`, it will be considered as multiple words split by space. In order to end a translation hypothesis, an end of sentence token "<\/s>" should be sent to the server.

---
## Get evaluation scores from the server
* **Method:** : `GET`

*  **Endpoint** : *server_url*/result

*  **URL Params:**

* **Success Response:**
  ```
  {
    "Quality": {
        "BLEU": BLEU
    },
    "Latency": {
        "AL": AL,
        "AL_CA": AL_CA,
        "AP": AP,
        "AP_CA": AP_CA,
        "DAL": DAL,
        "DAL_CA": DAL_CA
    }
}
  ```
Make sure to make this request after finishing all the translations.