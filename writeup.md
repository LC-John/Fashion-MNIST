# 图像分类模型的对抗攻击和对抗训练

## 对抗攻击（Adversarial Attack）

假设存在一图像模型（eg. 图像分类模型），可以接受图像输入，并产生正确的输出（eg. 预测分类）。攻击算法通过修改原本的图像数据，使得模型产生和原输出不同的输出。但是这样简单的修改非常容易被“识破”（例如把猫的图像直接换成狗，输入猫狗分类器，输出也一定就是狗）。

对抗攻击通过**扰动**原本的图像数据，即在仅对图像进行非常小的修改的条件下，使得模型产生和原输出不同的输出结果。这种扰动，一般情况下是人类难以发觉的，甚至是肉眼无法分辨的。对抗攻击产生的数据样本，一般具有如下特征：

1. 与原数据相比，变化极小
2. 一般情况下，扰动引起的差异会被人眼视为噪声，甚至人眼不可见
3. 会引起模型输出错误，但不会使得人类犯错

例如下面的一组例子。在一个mnist分类器上，上图的九个0均可以被正确识别为“0”。经过中图的扰动过程后，得到下图，此时这九个0从左到右会被该分类器依次识别为“1”到“9”。

<div style="text-align: center">
<img src="./images/demo_mnist_0.jpg"/><br>
<img src="./images/demo_mnist_adv.gif"/><br>
<img src="./images/demo_mnist_adv.jpg"/>
</div>

## 对抗攻击

对抗攻击有两种形式，即黑盒和白盒。

在黑盒攻击的场景下，整个待攻击模型都处于黑盒中，其模型框架、模型结构等所有超参数，以及模型各层权重等参数均不可知，仅有一个“predict”接口可用。该接口允许输入图像数据，并返回模型输出的各个类别的概率。除此之外的所有操作均被禁止。

在白盒攻击的场景下，整个待攻击模型均为透明，所有参数和超参数都可见，允许使用“predict”和“gradient”接口。“predict”接口与黑盒攻击中相同；“gradient”接口，允许输入图像数据和期望的输出，计算任意层上的（包括输入数据上的）的梯度并返回。除去训练和直接给权重赋值的操作，其余所有操作均被允许。

一般而言，图像模型的白盒攻击非常简单，而黑盒攻击相对困难，因为梯度信息可以明确指出扰动的方向；但在NLP领域中，黑盒攻击相对简单，而白盒攻击很困难，因为句子空间是离散空间，如何计算和利用梯度会很困难。

在现实场景中，黑盒攻击占据大部分，因而仅仅在“攻击”这一层面上，黑盒攻击更为重要。但是，白盒攻击可以针对模型产生特定的样本，这些对抗样本可以被用来修复模型原本的缺陷。因而就像“黑客”和“白客”，白盒攻击的目的并不是为了攻破模型，而是发现模型的缺陷，并辅助修复--因此在“防御”的层面上，白盒攻击更有意义。

## 白盒攻击 (White-box Attack)

分类器<img src="http://latex.codecogs.com/gif.latex?C" />接受输入图像<img src="http://latex.codecogs.com/gif.latex?x" />后产生各个类别概率向量<img src="http://latex.codecogs.com/gif.latex?\hat y" />，Loss function <img src="http://latex.codecogs.com/gif.latex?L(x,y|C)" />表示<img src="http://latex.codecogs.com/gif.latex?\hat y" />与图像<img src="http://latex.codecogs.com/gif.latex?x" />的真实类别<img src="http://latex.codecogs.com/gif.latex?y" />的距离。<img src="http://latex.codecogs.com/gif.latex?L(x,y|C)" />越小说明<img src="http://latex.codecogs.com/gif.latex?C" />的预测与真实的类别<img src="http://latex.codecogs.com/gif.latex?y" />越接近。进行攻击时，对<img src="http://latex.codecogs.com/gif.latex?x" />扰动得到<img src="http://latex.codecogs.com/gif.latex?\tilde x" />，使得<img src="http://latex.codecogs.com/gif.latex?L(x,\tilde y|C)" />尽量小，<img src="http://latex.codecogs.com/gif.latex?\tilde y" />即为攻击时的目标标签。因此白盒攻击其实是一个优化问题，即<img src="http://latex.codecogs.com/gif.latex?\min_{\tilde x}L(\tilde x,\tilde y|C)" />且<img src="http://latex.codecogs.com/gif.latex?\min_{\tilde x}dist(x,\tilde x)" />，该式表示期望<img src="http://latex.codecogs.com/gif.latex?\tilde x" />与<img src="http://latex.codecogs.com/gif.latex?x" />尽量相近，但模型输出<img src="http://latex.codecogs.com/gif.latex?C(\tilde x)" />为<img src="http://latex.codecogs.com/gif.latex?\tilde y" />。

介绍一个最简单的白盒攻击的梯度下降的方法，即“固定模型调输入样本”。

1. 初始化<img src="http://latex.codecogs.com/gif.latex?x^{(0)}=x" />；
2. 利用反向传播计算<img src="http://latex.codecogs.com/gif.latex?\\nabla_{x^{(n)}}L=\frac{\partial L(x^{(n)},\tilde y|C)}{\partial x^{(n)}}" />；
3. 利用梯度调节样本<img src="http://latex.codecogs.com/gif.latex?x^{(n+1)}=x^{(n)}-\alpha\cdot\nabla_{x^{(n)}}L" />，其中<img src="http://latex.codecogs.com/gif.latex?\alpha" />为学习速率；
4. 迭代直至<img src="http://latex.codecogs.com/gif.latex?argmax C(x^{(n)})=argmax \tilde y" />

这种方法相对简单粗暴，但并不能保证<img src="http://latex.codecogs.com/gif.latex?\tilde x" />与<img src="http://latex.codecogs.com/gif.latex?x" />尽量相近，产生出的<img src="http://latex.codecogs.com/gif.latex?\tilde x" />有可能会被人眼直接识别出来。更为高效的方法请参考相关论文。

## 黑盒攻击 (Black-box Attack)



## 对抗训练（Adversarial Training）

对抗训练时，使用白盒攻击，在训练集上产生一批对抗样本的数据，之后将这些数据重新掺入训练集中，利用新的数据集重新训练模型，从而可以“修复”模型原本的缺陷，达到提升性能以及抵抗对抗攻击的效果。

以图像二分类举例，简单说明对抗训练为什么效果会好。经过在训练集上若干轮的训练，分类器不断调整学习自己的各个参数，实质上表现为调整在图像超空间里的分类超平面，根据训练集的数据学习到的超平面是不可能和真实数据的分类超平面完全重合的。白盒训练得到的对抗样本的会略微跨过模型的分类超平面，而并未跨过真实数据的分类超平面（假设人眼分类就是真实数据的分类）。因此将这些对抗样本重新加入到训练集中，可以帮助修复模型的分类面偏离了真实数据的分类面的地方。

## 作业要求

### 数据集

使用Fashion MNIST数据集。这是一个类MNIST数据集，数据集的具体介绍请见```README.md```。

数据集位于```./data```子目录下，数据集的python3封装位于```./code/fmnist_dataset.py```文件中。

### 白盒攻击

对自己训练的模型进行定向白盒攻击，即原本被正确分类为A的图像，在扰动后被判别为指定类别B，A和B一一对应。要求：

1. 训练一个Fashion MNIST上的图像分类模型，要求分类准确率不低于95%；
2. 在test集中随机选出至少1000张可以被分类器正确分类的图像；
3. 对选出的图像进行白盒攻击，得到对抗样本，类别A和B的对应关系如下表。

| 原正确类别A    | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| ------------- | - | - | - | - | - | - | - | - | - | - |
| 指定的目标类别B | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 0 |

要求提交：

1. 分类器在test集上的准确率，以及分类器源码；
2. 白盒攻击的成功率，若攻击算法在迭代上限或运行时间上限前产生出可以使得分类器判别为类别B的样本，则认为成功，否则认为失败，上限阈值自定；
3. 在攻击成功的样本中随机选出10个，提交10张原图像和10张对抗样本图像，以及分类器判别的类别。

### 黑盒攻击

对提供的黑盒模型进行黑盒攻击，CNN模型代码位于```./code/model.py```，模型存放于```./model/model.ckpt```。要求：

1. 提供Fashion MNIST上的图像分类器，提供1000张待攻击的图像
2. 利用图像产生对抗样本，攻击提供的黑盒模型。
3. 黑盒模型不允许被打开，不允许被修改，不允许被用于白盒攻击。

可以考虑利用上一步的白盒模型和白盒攻击算法，产生对抗样本，并尝试将其迁移至该黑盒模型。也可以尝试其他的黑盒攻击算法，直接对黑盒模型进行攻击。

### 对抗训练

尝试对抗训练，提升模型性能和鲁棒性。要求：

1. 对抗训练得到新模型，测试其性能
2. 对新模型进行白盒攻击，测试是否获得抵抗白盒攻击的能力
3. 对新模型进行黑盒攻击，测试是否获得抵抗黑盒攻击的能力

可以直接利用白盒攻击得到的对抗样本进行对抗训练。也可以尝试利用其他对抗训练算法，进行对抗训练。
