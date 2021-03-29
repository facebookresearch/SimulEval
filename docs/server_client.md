# Server-Client interface

## Separate Server and Client
As introduced in [get started](./get_started.md),SimulEval uses a server-client structure to simulate the simultaneous translation setting.
![](architecture.png)

While a system can be simple evaluated use the command introduced in [get started](./get_started.md),
one can start a server process and client separately.
This is useful when debugging each part or evaluating multiple runs on a large dataset.
A stand-alone server command is
```
simuleval --server \
    --data-type text \
    --hostname localhost \
    --port 12345 \
    --source examples/data/src.txt \
    --target examples/data/tgt.txt
```
Notice that we have to set the `--data-type` because we are not able to infer the data type (text or speech) from the agent.
Once the server process start, we can kick off the evaluation by
```
simuleval --client \
    --agent examples/dummy_waitk_text_agent.py \
    --waitk 5 \
    --hostname localhost \
    --port 12345
```

On the other hand, if you have a server and a client command (let's say you were debugging the server),
a joined command can be achieved by merge all the arguments and remove `--server`, `--client`, `--data-type` (optional) arguments.

## Customized Client
While SimulEval client and agents follow the general logic for simultaneous translation, users can also implement their own client. The customized client can communicate with the server via a RESTful api, which can be found here. Here is
## **The structure of the evaluation client**
Here is example pseudocode for a client.
In practice, evaluation can be done in parallel
```
POLICY <- The function gives decision of read or write
MODEL <- The translation model

Start evaluation

N <- Request to get number of sentences in test set

Request to start a new evaluation session

For id in 0,..,N-1
    Do
        if POLICY is read
        Then
            Request to obtain a token or a speech utterence of sentence i
        Else
            W <- prediction of MODEL
            Request to send W of target sentence i to server
        EndIf
    While W is not <\s>

Request to get evaluation scores from server
```
