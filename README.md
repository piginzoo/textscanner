# 注意力模型识别OCR

这个是基于[@thushv89](https://github.com/thushv89/attention_keras)的注意力模型的基础上，添加了OCR的功能，为了表示尊敬，直接fork了他。

他对attention的实现的博客在[这里](https://towardsdatascience.com/light-on-math-ml-attention-with-keras-dc8dbc1fad39)。

但是，实际上我在他的基础上要添加OCR识别，参考的论文是
《Robust Scene Text Recognition with Automatic Rectification（RARE）》中的SRE网络。

![](./layers/images/sre.png)

# 算法实现思路

## 1.输入图像的特征化
本来打算直接用个vgg之类的backbone当做特征抽取的过程，即抽取后，reshape成所需要的序列输入的样子，比如7x7x512=>7x3584，结果后来了问题，虽然不用非要按照他的输入为224x224，而是就是直接resize成32高，恩，是的，高还是要统一的，然后输出的高度应该是Bx1xNx512，但是，N一般会很小，因为VGG都是宽高都缩成1/32，这个就不好办了，比如一个32x100的图片，缩成了1x3个feature map。3个，我怎么也不能搞出来5个字啊。你序列可以长点，但是你不能比要预测的还少啊。所以，只好放弃vgg了。

然后只好用crnn项目中卷积网络，那个卷积，宽只会变成1/4，而高可以变成1/32，这样就可以变成一个 [1, W/32, 512]的一个特征序列了，然后就可以灌给Bi-GRU了。

## 2.Bi-GRU Encoder

这个没啥，就是一个标准的双向GRU编码器，注意这个属性就可以：`return_sequences=True`，毕竟要返回一个完整的序列，而不是仅仅最后一个状态。

## 3. GRU解码器

注意他的隐含层是编码器的2倍这个细节，毕竟Bi-GRU是前向和后向concat一起的结果嘛。

## 4.注意力编码器了

是[@thushv89](https://github.com/thushv89/attention_keras)开发的注意力模型，细节上也没啥，主要是里面用了两个k.rnn，关于K.rnn函数，读代码时候会遇到，参考[这个](https://kexue.fm/archives/5643/comment-page-1)，其实核心就是把一个序列生成事儿，转化成一个循环执行的函数了。代码也不难，细看也容易明白，就不赘述了。

![](./layers/images/attention.png)

## 5. 最后的输出

最后呢，是把attention输出的内容和解码器GRU的输出，concat一起，扔给一个TimeDistributed，其实就是一个共享的全连接，输出一个3770（词表）大小的分类就完事了。

# 设计中的纠结和问题

最后的实现，看上去简单简洁，其实经历过多次的纠结和趟坑，记录下来，害怕忘却。

## 思路转变的痛苦

说说我的整个思考历程，

最开始，就是觉得照着SRE的网络结构撸就可以，找到了thushv89的注意力模型，就想，前面接一个vgg，做特征抽取，灌给一个seq2seq，然后结合上thushv89的注意力，完事。

开始，就是vgg，VGG的放弃，就不多说了，上面谈到了已经。就是因为宽度不够。只好换成了[crnn](https://github.com/piginzoo/crnn)中的Conv卷积网络，得到了一个[1,Width/4,512]的序列。

然后，就交给attention+seq2seq吧，也没觉着多难，人家thushv89给的例子中，有代码啊，于是拷贝过来，跑了一一下，虽然也遇到问题，不过，还是通了。

然后，就开始纠结mask问题了。

输入的时候，虽然我把图片都resize成高度为32，宽度依据高度调整自动调整了，但始终是不一样的，之前在我的[crnn](https://github.com/piginzoo/crnn)中，由于我解决不了卷基层可以支持变长sequence的问题，我就改用保持形状的padding（即高度32，宽度调整后，后面加padding），这次，我用keras实现，发现keras里面有个masking层，于是觉得，是不是可以试试呢。

后来发现，需要mask的玩意有2个：
- 图片，准确说，是最开始卷积开始的时候就需要masking；然后卷积完的特征序列，还需要mask后，才好意思灌给bi-gru
- 文字，被识别的文字也是参差不齐的，也需要masking呀，过去在crnn里面，用tensorflow的**sequence_length**,在keras对应就是masking。

```
rnn.stack_bidirectional_dynamic_rnn(fw_cell_list, bw_cell_list, inputdata,sequence_length=sequence_len,dtype=tf.float32)
```

于是，尝试用masking，

首先是卷积层根本就不支持masking，所以，一开始的input就masking之的幻想破灭。

然后，那我就尝试，卷积之后用个lambda来做padding吧，然后尝试套个mask给他：
```bash
def padding_wrapper(conv_output):
    paddings = [[0, 0], [0, 50 - tf.shape(conv_output)[0]],[0,0]]
    conv_output_with_padding = tf.pad(conv_output, paddings=paddings)
    conv_output_with_padding.set_shape([None, 50, 512])  # 靠！还可以这么玩呢！给丫设置一个shape。
    return conv_output_with_padding
    
conv_output_with_padding = Lambda(padding_wrapper)(conv_output);
```
这个算是过了，可是，再往后，这玩意还要往后传，传到attention里面以后，attention里面又报错，说不支持masking了。

后来，我理解，masking的本意是用一个特殊的值来标识，这些值不参与运算，用来解决类似于seqence长度不一的情况，但是，这个masking一旦某一层开始实施，就得要求后续的层也需要支持，一般都是要在构造函数里面加上`self.supports_masking = True`，还要在call的时候，加入入参mask等等。

也就是说，我还要调整attention的相关代码，来适应masking，而这个调整动静有些大，我需要修改各个k.rnn的能量函数啥的，都要改，我权衡了一下，算了，放弃了。

最后，我又回到老路上，老老实实的定义一个最大宽度，然后加如padding，而且还是用的标准的tensorflow的tf.pad_seqence的方式。

## 总结一下整体思路

就是一个CNN+一个Seq2Seq，CNN是一个标准的CNN，只是用Keras的Layer封装了一下而已，不过这里是有问题的，下节说。

然后是Seq2Seq，是一个双向GRU+一个GRU，编码器是双向GRU，解码器是一个单向GRU。

训练的时候，需要的输入是两个，一个是图片，一个是需要解码的字符串，但是以STX开头。标签是1个，是需要解码的字符串，但是是以ETX结尾的。这个是标准的训练方式，即，解码器的输入是标签的前一个字。需要注意的是，在真正预测的时候，就需要使用前一个时间输出的字符了，这个体现在模型上，就是训练模型和预测模型是不一样的，但是参数是相同的。

Attention的代码仔细看了一遍，没有任何问题，不过细节确实很多，需要很认真的捋一遍才能理解，特别是对于K.rnn函数的理解，以及在这个attention代码中的应用，要尤其注意。

Attention是在编码和解码都完成之后进行的，用编码器和解码器的输出，共同算出了编码器的每个步骤的加权平均结果，再灌入解码器的每个时间片的最后一步，即全连接+softmax，得到最后的解码结果。

词表中加入了3个额外的词，分别是0：空格；1：ETX；2：STX；分别用来做词的padding，句子结束标志，句子开始标志。


# 遇到的版本的坑

## 坑1：不要混用keras和tf.keras的代码

争取的代码为：
```
from tensorflow.python.keras.models import Model
from layers.attention import AttentionLayer
from tensorflow.python.keras.utils import to_categorical
```

如果混用，如下代码
```
from keras.layers import Bidirectional,Input, GRU, Dense, Concatenate, TimeDistributed,Reshape
from tensorflow.python.keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import VGG19
```

就会报错：
 ```
 AttributeError: 'Bidirectional' object has no attribute 'outbound_nodes'
 ```
 
原因是不能用keras自带的vgg19+keras自带的bidirectional，靠，肯定是版本不兼容的问题切换到下面的就好了，之前还是试验了用tf的bidirectional+keras的vgg19，也是不行，报错：AttributeError: 'Node' object has no attribute 'output_masks'

靠谱的组合是：tf的bidirectional+tf的vgg19

```
from tensorflow.python.keras.applications.vgg19 import VGG19
from tensorflow.python.keras.layers import Bidirectional,Input, GRU, Dense, Concatenate, TimeDistributed,Reshape
```
是的，全部都替换成tensorflow里面的keras的东东就可以了。

## 坑2：seq2seq中的坑

seq2seq的实现中，也有很多细节的坑

- masking，不是说，不计算后面的值了，而是后边的输出值不变了。比如：
```
test_input = np.array([
    [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
    [[2, 2, 2], [2, 2, 2], [0, 0, 0], [0, 0, 0]],
    [[3, 3, 3], [0, 0, 0], [0, 0, 0], [0, 0, 0]]], dtype=int)
mask = Masking(mask_value=0)
gru = GRU(
    1,
    return_sequences=True,
    activation=None,
    recurrent_activation=None,
    kernel_initializer='ones',
    recurrent_initializer='zeros',
    use_bias=True,
    bias_initializer='ones'
)
x = Input(shape=test_input.shape[1:])
m1 = Model(inputs=x, outputs=gru(x))
print(m1.predict(test_input).squeeze())
结果：
[[  -12.   -60.  -252. -1020.]
 [  -42.  -336.  -336.  -336.]
 [  -90.   -90.   -90.   -90.]]
``` 
看，mask起作用了，看来后面的就没有再计算了，依然保持输出值。

- 这个seq2seq和注意力的结合中并没有再次输入attention结果到解码GRU的输入，而是直接和解码器GRU的输出concat到一起，然后就做全链接分类了。

# 实现中的trick

谈谈具体实现中的一些细节，

## 训练数据的加载

这次尝试了Keras的Sequence，我们都知道，Keras的数据输入有3中方式，一种是全部加载全体数据；一种是ImageGenerator；还有一个Sequence类的实现。

对应到训练的时候，又可以对应fit、fit_generator和train_on_batch，该如何选择呢呢？
[这篇文章](https://www.twblogs.net/a/5c226c9fbd9eee16b3db025b)讲的明白：
>在99％的情況下，您不需要對訓練深度學習模型進行如此精細的控制。相反，您可能只需要自定義Keras .fit_generator函數。

恩，我就用fit_generator了。

另外，ImageGenerator不是为了加载数据用的，是为了做数据增强用的，虽然可以那么用，
另外，在model.fit_generator里面可以用迭代器，但是无法迭代器方式无法开启多进程，
最佳姿势是：Sequence+多进程：
```python
D = SequenceData('train.csv')
model_train.fit_generator(generator=D,
    steps_per_epoch=int(len(D)), 
    epochs=2, 
    workers=4, 
    use_multiprocessing=True, 
    validation_data=SequenceData('vali.csv'),
    validation_steps=int(20000/32))  
```

在构造函数里面，调用initialize方法，完成整体图片文件名的加载，（不加载数据），然后在__getitem__方法，一批批的从磁盘上读取数据，而且，框架帮我控制内部的进程数，很好很好，我很满意。

## seq2seq的数据准备

谈谈标签数据吧，就是一个个的字符串嘛，但是，对seq2seq的解码器GRU来说，他的输入和输出是啥呢？

输入是一个字符串，开始是BOS，然后是第一个字符，第二个字符。。。。
输出是一个字符串，和输入一样的，但是，但是，第一个输出没有BOS，而是从第一个字符开始的，最后一个输出是EOS。

所以，需要在绑定输入和输出的时候，调整一下：
```python
model = Model(inputs=[input_image, decoder_inputs], outputs=decoder_pred)
--------------------------------
return [images,labels[:,:-1,:]],labels[:,1:,:]
```

另外，由于训练的数据量非常大（上百万的样本量），要进行这个预处理很费时间，因此增加了一个参数叫"preprocess_num"，来启动对应数量的进程，同时完成预处理，提高加载速度。

## 训练细节

```python
    model.fit_generator(
        generator=train_sequence,
        steps_per_epoch=args.steps_per_epoch,#其实应该是用len(train_sequence)，但是这样太慢了，所以，我规定用一个比较小的数，比如1000
        epochs=args.epochs,
        workers=args.workers,
        callbacks=[TensorBoard(log_dir=tb_log_name),checkpoint,early_stop],
        use_multiprocessing=True,
        validation_data=valid_sequence,
        validation_steps=args.validation_steps)
```
- 一个是generator是自定义的sequence，前面已经详细介绍过了
- steps_per_epoch默认为

# 开发日志

## 2019.8.21 

修正一下bugs:

- 数据加载存在bug，有none数据加入，剔除了他们
- padding逻辑修正为，如果不到200像素宽，就加**"白色"**来padding，之前是黑色；超过200就resize成200（会导致变形，但也比截取掉强）
- sequence的on_epoch_end中shuffle动作，仅shuffle indices，而不是之前shuffle整个数据数组
- 调整了训练的参数，并添加了关键参数的注释

增加一些新特性：

- 增加了一个加载checkpoint，可以继续训练

依然存在的问题：

训练过慢，目前看1000个batch下来要13分钟左右，很慢，之前300万的数据大于是5万个batchs，会非常慢，
所以加入了steps_per_epoch来调整为1000个batch作为一个epochs。
这样做，会有一些副作用，就是每次只能取1000个batch数据训练，然后就要shuffle整个数据集。
为何要把epochs缩短，原因是Keras是在每个epochs结束的时候，才会回调诸如validate、early stop、checkpoint等回调。
目前也只有这个解决方案了。

## 2019.8.22

发现一个新问题，在生产环境出现了sequence加载数据卡住的情况，于是经过本地的实验和测试，发现：
- 不用用多进程方式加载，不知道为何总是卡住，改成multiprocess=False,即使用多线程就好了，[参考](https://stackoverflow.com/questions/54620551/confusion-about-multiprocessing-and-workers-in-keras-fit-generator-with-window)
- on_epoch_end确实是所有的样本都轮完了，才回调一次，而，steps_per_epoch改变的是多久callback回调一次，这个可以调的更小一些，两者没关系
- 修改了CNN网络中的batch normalization部分，之前的方法不对
- 解决了之前预加载checkpoint模型无效的bug，之前的model.load_weights无效，采用tensorflow.python.kears.models.load_model来加载
- 自定义了各类自定义对象，替换了默认的accuracy，以及各类自定义layer，否则，load_model会报错：
```python
     model = load_model(_checkpoint_path,
            custom_objects={
             'words_accuracy': _model.words_accuracy,
             'Conv':Conv,
             'AttentionLayer':AttentionLayer})
```
- 另外装在checkpoint的时候，遇到一个警告："WARNING:tensorflow:Layer decoder_gru was passed non-serializable keyword arguments: {'initial_state': [<tf.Tensor 'concatenate_1/concat:0' shape=(?, 128) dtype=float32>]}. They will not be included in the serialized model (and thus will be missing at deserialization time)."，
[网上](https://github.com/keras-team/keras/issues/9914)也有人遇到，但是貌似也没啥解决办法，也不太了解影响，所以我也暂时忽略了。

## 2019.8.23
- 增大了GRU神经元数量64=>512



## 2019.12.29

之前训练一直不收敛，但是一直也没有特别认真的把代码再捋一遍，最近，在ZN的帮助下，我们一起把代码捋了一遍，终于发现了一些问题。

由于代码已经是4、5个月之前写的，难免都有些遗忘了，我先把整儿思路捋一遍，然后再说我代码的问题。

还是发现了一些问题：

主要是修改卷基层，和之前的CRNN的代码对比了一下，发现了不少问题，后来照着CRNN的代码重新改了一遍。

发现的主要问题是，BatchNormal应该是是在激活函数Relu之后。

## 2020.1.2

之前的sequence是有问题，一次都加载到内存里了，其实是误解了sequence的用法了。

正确的姿势是，在__init__中只要告诉全部数据的条数即可，在__getitem__里面才真正去加载文件和做预处理呢，idx还是标明批次的。

```python
   model.fit_generator(
        generator=train_sequence,
        steps_per_epoch=args.steps_per_epoch,#其实应该是用len(train_sequence)，但是这样太慢了，所以，我规定用一个比较小的数，比如1000
        epochs=args.epochs,
        workers=args.workers,   #<------------- 同时启动多少个进程加载
        callbacks=[TensorBoard(log_dir=tb_log_name),checkpoint,early_stop],
        use_multiprocessing=True, #<----------- 这里开启多进程，就可以多进程同时处理样本加载了，内部会有一个queue来缓存
        validation_data=valid_sequence,
        validation_steps=args.validation_steps)
```

## 2020.1.3

遇到一个诡异的异常：

发生在训练的时候，看字面意思是，说attention_layer/U_a这个参数未被初始化，诡异的是，说Adam的ReadVariableOp操作CPU中变量attention_layer/U_a，
可是，这个变量明明在GPU中呢

```python
Traceback (most recent call last):
  File "/usr/lib/python3.5/runpy.py", line 184, in _run_module_as_main
    "__main__", mod_spec)
  File "/usr/lib/python3.5/runpy.py", line 85, in _run_code
    exec(code, run_globals)
  File "/app.fast/projects/attention_ocr/main/train.py", line 100, in <module>
    train(args)
  File "/app.fast/projects/attention_ocr/main/train.py", line 86, in train
    validation_steps=args.validation_steps)
  File "/root/py3/lib/python3.5/site-packages/tensorflow/python/keras/engine/training.py", line 1761, in fit_generator
    initial_epoch=initial_epoch)
  File "/root/py3/lib/python3.5/site-packages/tensorflow/python/keras/engine/training_generator.py", line 190, in fit_generator
    x, y, sample_weight=sample_weight, class_weight=class_weight)
  File "/root/py3/lib/python3.5/site-packages/tensorflow/python/keras/engine/training.py", line 1537, in train_on_batch
    outputs = self.train_function(ins)
  File "/root/py3/lib/python3.5/site-packages/tensorflow/python/keras/backend.py", line 2897, in __call__
    fetched = self._callable_fn(*array_vals)
  File "/root/py3/lib/python3.5/site-packages/tensorflow/python/client/session.py", line 1454, in __call__
    self._session._session, self._handle, args, status, None)
  File "/root/py3/lib/python3.5/site-packages/tensorflow/python/framework/errors_impl.py", line 519, in __exit__
    c_api.TF_GetCode(self.status.status))
    tensorflow.python.framework.errors_impl.FailedPreconditionError: 
    Error while reading resource variable attention_layer/U_a from Container: localhost. 
    This could mean that the variable was uninitialized. 
    Invalid argument: Trying to access resource located 
        in   device /job:localhost/replica:0/task:0/device:GPU:0 
        from device /job:localhost/replica:0/task:0/device:CPU:0
	 [[Node: training/Adam/ReadVariableOp_86 = ReadVariableOp[dtype=DT_FLOAT, 
	    _device="/job:localhost/replica:0/task:0/device:CPU:0"](attention_layer/U_a/_235)]]
```
开始以为是之前的keras和tf.keras的问题，检查了后，没有发现混用的地方的。我改成了tensorflow.keras.xxx => keras.xxx，结果出现了上述问题，唉，很诡异，肯定是没啥问题的啊，
然后我就查，发现，Conv，也就是卷积那块，没有做compute_output_shape的，输出shape的计算，加上了，还是不行，
没办法，只好回滚，从 keras.xxx⇒tensorflow.keras.xxx，错误依旧，还是Adam的时候GPU引用了CPU之类的东西，
记得之前google的时候，很多帖子都是说keras版本的问题，我check了我的keras是2.1.0，tensorflow是tensorflow-gpu==1.8.0，没问题啊。
最后，彻底放弃，直接查了一个我当前CUDA9能支持的最大tensorflow==1.12.0，而keras是2.2.0，安装之，死马当活马医了，结果，好了！
结论，此坑太深，之前看了不下30片帖子，都没有确切的线索，好几篇都是提到了keras的和tensorflow.keras的混用问题，以及版本问题，之前还试验过1.9，但是，最终还是极端的换成了1.12，才解决。

但是，跑了1天1夜，还是不收敛，loss的下降速度很慢，正确率也上不去。继续观察。

# 如果需要跑原作者的例子

作者的代码目前被转移到test/examples目录下了。

为了加深attention的理解，可以跑原作者的Attention的例子：

`python -m examples.nmt_bidirectional.train`

要跑的话，需要先准备数据：
```
cd data
tar -xvf small_vocab_en.txt.gz
tar -xvf small_vocab_fr.txt.gz
```
