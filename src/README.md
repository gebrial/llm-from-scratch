This folder contains scripts testing and improving the gpt trainer (which is in the components folder)

### todo:
* ~~move dataloader setup into lightning module~~ (done)
* ~~use setup() function in lightning module to setup model~~ (done)
* use/test flash attention in attention module to increase speed. head dimension must be one of {16, 32, 64, 128}? context length must be multiple of 64?
* **try pre-training on simple wikipedia dataset**
* ~~use torch.compile to increase training speed~~ (done)
* ~~modify accuracy function to not count "endoftoken" tokens~~ (using sequence packing makes this unnecessary)
* ~~investigate mem leak by implementing bare bones training loop~~ (caused by multrithreading, copy on read issue)
* ~~process text data into files with tokens and save metadata for use later (e.g., total tokens, total stories, etc.)~~ (using huggingface datasets makes this unnecessary)
* use lower memory consumption data types for inputs (e.g., int16)
* ~~confirm that attention masks are being used properly. API says error should be thrown when mask and is_causal=True is set, but no error is thrown.~~ (confirmed working. error is thrown when mem_efficient_sdp is disabled. otherwise it applies both causal masking AND the mask applied meaning the only values attended to are the ones that neither mask blocks)
* try using triton for cross entropy loss calculation (torchtune version shows no improvement over nn functional version in speed, cpu mem usage, or gpu mem usage) (couldn't figure out what the inputs for the apple cce implementation are supposed to be)
* ~~configure positional encodings so that they match the story positions in the packed sequences (implemented, need to train)~~ (done)
* ~~use smaller vocab to train faster with more layers (generated small vocab tokenizer, need to process text and train with it)~~ (done, less weird symbols but still finding random "a" characters here and there, probably from accent remover normalizer)
* investigate batches where loss explodes
* ~~clean dataset further (completely remove any stories with weird characters)~~ (done)
* process inputs so that end of sequence doesn't predict beginning of next sequence (it should predict another endoftoken)
* resubmit prompt completions with fixed version of text generator (this time don't skip special tokens like \n)
* train on highest quality stories at the end of training run (maybe longest stories less than 512 tokens or use gpt model to rate stories and use highest rated stories)
* ~~use beam search to generate top 5 results~~ (done, all beams will generate almost exactly the same result with variations only in the last few words of the output)
* ~~use top-p sampling with temperature to generate output~~ (done, works much better than beam search)

### resolve:
* importing in scripts made from notebook in different folder

### Tips:
* use huggingface datasets to download/process datasets (very fast + memory efficient)
* to properly save/load models using pytorch lightning, ensure that model is initialized in configure_model() function in LightningModule
* compile model to reduce training time
* in pytorch lightning, in the configure_model hook, make sure you check if self.has_attr("model") before configuring a new model, otherwise it will override any existing model (e.g., when loading in a model to a trainer just to validate)


