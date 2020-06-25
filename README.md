This is the [Text Scanner paper](https://arxiv.org/abs/1912.12422) implementation code.

About the paper unstanding, please read about it at my blog : [TextScanner的一些研究](http://www.piginzoo.com/machine-learning/2020/04/14/ocr-fa-textscanner) 

# implementation list(ongoing...)
- [X] implement the network
- [ ] implement the mutual-supervision mechanism
- [X] implement loss function
- [X] create the character annotation GT, and prepare none character level GT
- [X] implement train code
- [ ] implement evaluation code
- [ ] train the model

# developing logs
- 2020.4.24 create the project and implement the skeleton of the project
- 2020.4.30 implement the network code, and finish the GT generator and loss function
- 2020.5.12 the network works now after hundreds of trouble-shootings,TF2.0/tk.keras is full of pit
- 2020.6.03 make a new branch to solave the OOM issue

# implement details
Developing detail can be tracked by my [textscanner implementation issues](https://www.notion.so/piginzoospace/Textscanner-254a700668714f0d811afe2ab8124046).