This folder contains scripts testing and improving the gpt trainer (which is in the components folder)

### todo:
* ~~move dataloader setup into lightning module~~ (done)
* ~~use setup() function in lightning module to setup model~~ (done)
* use/test flash attention in attention module to increase speed. head dimension must be one of {16, 32, 64, 128}? context length must be multiple of 64?
* try pre-training on simple wikipedia dataset
* use torch.compile to increase training speed
* ~~modify accuracy function to not count "endoftoken" tokens~~ (using sequence packing makes this unnecessary)
* ~~investigate mem leak by implementing bare bones training loop~~ (caused by multrithreading, copy on read issue)
* ~~process text data into files with tokens and save metadata for use later (e.g., total tokens, total stories, etc.)~~ (using huggingface datasets makes this unnecessary)
* use lower memory consumption data types for inputs (e.g., int16)
* ~~confirm that attention masks are being used properly. API says error should be thrown when mask and is_causal=True is set, but no error is thrown.~~ (confirmed working. error is thrown when mem_efficient_sdp is disabled. otherwise it applies both causal masking AND the mask applied meaning the only values attended to are the ones that neither mask blocks)
* try using triton for cross entropy loss calculation (heavy memory usage with torch implementation?)
* configure positional encodings so that they match the story positions in the packed sequences

### resolve:
* importing in scripts made from notebook in different folder
