#!/usr/bin/env python
# coding: utf-8

# This folder contains scripts testing and improving the gpt trainer (which is in the components folder)

# ### todo:
# * ~~move dataloader setup into lightning module~~ (done)
# * ~~use setup() function in lightning module to setup model~~ (done)
# * use/test flash attention in attention module to increase speed
# * try pre-training on simple wikipedia dataset
# * use torch.compile to increase training speed
# * modify accuracy function to not count "endoftoken" tokens
# * 
