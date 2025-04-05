This folder contains scripts testing and improving the gpt trainer (which is in the components folder)

### todo:
* ~~move dataloader setup into lightning module~~ (done)
* ~~use setup() function in lightning module to setup model~~ (done)
* use/test flash attention in attention module to increase speed
* try pre-training on simple wikipedia dataset
* use torch.compile to increase training speed
* ~~modify accuracy function to not count "endoftoken" tokens~~ (using sequence packing makes this unnecessary)
* ~~investigate mem leak by implementing bare bones training loop~~ (caused by multrithreading, copy on read issue)
* ~~process text data into files with tokens and save metadata for use later (e.g., total tokens, total stories, etc.)~~ (using huggingface datasets makes this unnecessary)

### resolve:
* importing in scripts made from notebook in different folder
