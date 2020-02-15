# VGG

Status: finished
URL: https://arxiv.org/abs/1409.1556

# Very Deep Convolutional Networks for Large-Scale Image Recognition

# Abstract

- CNN의 depth가 accuracy에 어떤 영향을 미치는지 investigate
- 네트워크의 depth를 늘렸을 때 significant한 성능 향상이 이루어짐

# **Introduction**

- 최근 Computer vision의 다양한 문제에서 CNN이 높은 성능을 보여주고 있다.
- 해당 논문에서는 CNN 아키텍처를 설계하는 데 있어서 depth의 중요성에 초점을 맞춘다.
- 다른 파라미터들은 그대로 두고 3x3 Conv 레이어만 늘리면서 CNN의 depth에 따라 성능이 어떻게 달라지는지 확인한다.

# ConvNet Configuration

- 3x3 conv를 사용한다. Why?
    - left, right, up, down, center 방향을 모두 capture할 수 있는 가장 작은 사이즈의 커널이기 때문

        ![VGG/Untitled.png](VGG/Untitled.png)

    - 1x1 conv를 사용하는 버전도 있음 1x1 conv에서 채널 수를 변경하지 않으며, 입력 channel의 linear transformation으로 볼 수 있음
- padding, stride를 1로 주어, 모든 conv layer에서 spatial resolution이 보존됨
- spatial size는 2x2 max pooling(stride=2)에서 줄어듦
- 마지막 conv, max-pool 레이어 이후 3개의 f.c레이어 사용 4096, 4096, 1000
- activation function ReLU사용,
- Local Response Normalization 사용x
    - LRN레이어가 이미지넷 데이터셋에서 성능 향상이 되지 않음
    - 메모리 사용량이랑 computation time 증가시키는 단점 있음

### Architecture

![VGG/Untitled%201.png](VGG/Untitled%201.png)

![VGG/Untitled%202.png](VGG/Untitled%202.png)

- layer가 많은 네트워크일지라도 파라미터 수 차이가 크게 나지 않는다.

## Discussion

VGG의 네트워크 구조가 기존의 ImageNet 2012, 2014 sota였던 AlexNet, ZF-Net와 다른 점이 무엇인가?

- 11x11, 7x7의 large receptive field를 사용했던 것에 비해 상대적으로 매우 작은 크기의 3x3 receptive filed를 사용하였다.
- 5x5 conv는 3x3 conv 두 번 사용한 것과 같은 receptive filed를 가지며, 7x7 conv는 3x3 conv를 세 번 사용한 것과 같은 receptive filed를 가진다.

![VGG/Untitled%203.png](VGG/Untitled%203.png)

[https://www.researchgate.net/figure/The-receptive-field-of-each-convolution-layer-with-a-3-3-kernel-The-green-area-marks_fig4_316950618](https://www.researchgate.net/figure/The-receptive-field-of-each-convolution-layer-with-a-3-3-kernel-The-green-area-marks_fig4_316950618)

- 그럼 그냥 5x5, 7x7 한 번만 쓰면 되지 뭐하러 3x3 여러번 사용하는가?
    - 비선형 함수를 더 많이 사용하여 decision function이 더 discriminative해진다
    - 1x1 conv를 사용하는 것 역시 receptive filed에 영향을 주지 않고 비선형성을 증가시킬 수 있는 방법이라 할 수 있음
    - 파라미터 수를 줄일 수 있다.
        - input & output 채널 수가 같은 3x3 conv를 3번 사용했을 때 파라미터 수는 3x(3x3xCxC)가 된다.
        - 동일한 조건으로 7x7 conv를 한번 사용했을 때 파라미터 수는 7x7xCxC가 된다.

⇒ Small size의 conv filter를 사용하는 conv layer를 여러 개 쌓는 것이 성능을 향상시킬 수 있다.

# Evaluation

- single scale을 사용한 evaluation으로 train image와 test image의 size가 고정되어 동일한 경우

![VGG/Untitled%204.png](VGG/Untitled%204.png)

- Local Response Normalization을 사용했을 때 오히려 error rate이 더 높았다. 이로 인해, 모델 B부터는 LRN을 적용하지 않았다.
- 네트워크의 depth가 깊어질수록(layer 수 증가) 성능이 향상되었다.
- C(use 1x1), D(only 3x3)는 같은 depth를 가지지만 D가 성능이 더 높았다.
    - additional non-linearity가 성능 향상이 더 도움이 되기 때문
    - 1x1보다는 3x3가 spatial context 잘 capture할 수 있기 때문에 더 중요하게 작용함
- 19개의 레이어를 사용했을 때 error rate saturate되었음, 그러나 더 large한 dataset의 경우 depth가 더 깊은 모델을 사용하는 것이 유리할 수 있음
- 기존 모델 B에서 3x3 conv 2개를  5x5 1개로 바꿔봤음, 그러나 오히려 error가 더 높게 나옴. 즉, small filter를 사용하는 것이 더 outperform
- training image에 scale jittering을 사용했을 때 성능이 더 높았음
    - scale jittering이란 이미지의 크기를 256 or 384 같은 고정된 크기로 조절하거나 [256; 512] 같은 특정 범위의 사이즈로 조절한 뒤 CNN의 입력 사이즈(224x224)에 맞게 무작위로 cropping하는 것

- 이번에는 test image에 scale jittering을 사용하여 multi scale image에 대한 evaluation

    ![VGG/Untitled%205.png](VGG/Untitled%205.png)

- train image와 test image간의 scale이 너무 차이나면 오히려 성능이 떨어질 수 있으므로 크게 차이나지 않는 범위에서 scale을 조정함
- train image = Q, test image = S이며 Q=S라고 했을 때, Q = {S-32, S, S+32}의 3가지 scale로 test image의 사이즈를 조절함
- Table 4에서 볼 수 있는 것처럼 multiple scale test image를 사용했을 때 성능이 증가함
- Table 4에서 모델 D의 마지막 결과를 보면 training & testing 과정에서 scale jittering을 적용한 것이 가장 성능이 좋았음. 왜냐하면 네트워크가 testing에서 들어오는 다양한 scale의 이미지를 트레이닝 과정에서 학습할 수 있기 때문

- 지금까지의 실험에선 단일 모델을 가지고 평가했지만 이번에는 여러 모델의 output을 combine하여 평가함. combine 과정은 단순하게 각 class에 대한 확률값을 평균내어 사용함

    ![VGG/Untitled%206.png](VGG/Untitled%206.png)

- Table 5에서 second row를 보면 scale jittering이 서로 다른 D 모델 3개를 조합함. 그 결과 단일 모델로 D를 사용했을 때 25.9%, 8.0%의 error를 냈던 결과에 비해 25.3%, 7.8%로 더 높은 성능을 냄
- 7개의 모델을 combine 했을 때  error rate이 더 줄어듦
- 7개의 모델이 내는 예측 값을 단순하게 평균낸 것이 아니라 각 모델마다 가중치를 두고 반영함. 어떤 모델의 신뢰도가 더 높은지 평가하는 기준은 top-5 validation error를 보고 판단함. 모델 C보다는 모델 E가 성능이 좀 더 좋다고 보기 때문에 모델 C와 E의 가중치를 서로 다르게 두는 것.

- ILSVRC-2014 classification result

![VGG/Untitled%207.png](VGG/Untitled%207.png)

- 제안하는 네트워크가 single net 기준으로 SOTA
- 이전 우승자들의 성능을 모두 이김
- winner인 GoogleNet과 비교했을 때도 competitive하며 다른 method보다는 outperform
- VGG Net을 설계할 때 classic한 CNN 아키텍처에서 시작한 것이 아니라 네트워크의 depth를 어디까지 증가시킬 수 있느냐에 관점을 두고 설계함
- VGG 1 net에서 test error 없는 이유: 시간부족으로 인해 best performance single network를 deadline까지 제출하지 못 함

# Localization

- classification task가 아닌 localization task에 적용했을 때의 결과
- localization이란? 이미지 내에 1개의 object만 존재하며 bounding box로 object의 위치를 localization해야하고 해당 이미지가 어떤 class인지 prediction하는 문제

### Localization ConvNet

- 기존에 class score를 예측했던 last [f.c](http://f.cl) layer가 bounding box location을 예측하도록 바꿔야 함
- classification task에서 제일 성능이 좋았던 아키텍처 D를 사용함
- classification CNN과의 가장 큰 차이점은 logistic regression objective(분류 문제를 푸는 모델)를 Euclidean loss로 변경. (Ground-truth와 예측 bounding box 사이의 loss를 구하기 위해)
- 2개의 localization model을 사용하였으며 각각 256x256, 384x384 size로 트레이닝 함. classification model에서 사용된 pre-trained 모델을 사용하기 때문에 별도의 training scale jittering은 사용하지 않음
- 모든 layer를 fine-tuning하는 것과 2개의 first two f.c layer만 fine-tuning하는 두가지 방법을 고려함. 마지막 f.c layer는 랜덤하게 초기화하고 train from scratch

- SCR(single-class regression), PCR(per-class regression)

![VGG/Untitled%208.png](VGG/Untitled%208.png)

- first two f.c layer를 fine-tuning하는 것보다 모든 레이어를 fine-tuning하는 것이 더 좋음

- multiple scale image로 학습하고 테스트 하는 것이 더 성능이 높다.

    ![VGG/Untitled%209.png](VGG/Untitled%209.png)

- localization task에서 VGG를 다른 모델과 비교

    ![VGG/Untitled%2010.png](VGG/Untitled%2010.png)

- top-5 test error가 25.3%로 최고 성능을 달성함.

- 다양한 image classification 데이터셋에서 SOTA를 찍었던 모델들과의 성능 비교

![VGG/Untitled%2011.png](VGG/Untitled%2011.png)

- *(asterisk) 달린 모델은 class 2000개짜리 ILSVRC dataset으로 pre-training시킨 모델
- 기존 모델들과 비교했을 때 VOC-2007, VOC-2012, Caltech-256 dataset에서 기존 SOTA를 달성했던 모델보다 더 높은 성능을 보여주었으며 Caltech-101 dataset의 경우에도 competitive한 결과를 보여줌

# Conclusion

- large scale image classification에서 기존의 CNN보다 레이어를 더 깊게 쌓아 성능을 향상시킴(2위)
- Object localization challenge에서도 SOTA
- 네트워크의 depth는 classification accuracy에 beneficial한 요소이며 visual representation 측면에서도 매우 중요함

    import tensorflow as tf
    
    
    
    class VGG16(tf.keras.Model):
        def __init__(self, nb_classes):
            super(VGG16, self).__init__()
    
            self.nb_class = nb_classes
            
            self.conv1_1 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')
            self.conv1_2 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')
            self.max_pool1 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))
            
            self.conv2_1 = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')
            self.conv2_2 = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')
            self.max_pool2 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))
    
            self.conv3_1 = tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')
            self.conv3_2 = tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')
            self.conv3_3 = tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')
            self.max_pool3 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))
    
            self.conv4_1 = tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu')
            self.conv4_2 = tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu')
            self.conv4_3 = tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu')
            self.max_pool4 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))
    
            self.conv5_1 = tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu')
            self.conv5_2 = tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu')
            self.conv5_3 = tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu')
            self.max_pool5 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))
    
            self.flat = tf.keras.layers.Flatten()
            self.dense1 = tf.keras.layers.Dense(4096, activation='relu')
            self.dense2 = tf.keras.layers.Dense(4096, activation='relu')
            self.dense3 = tf.keras.layers.Dense(nb_classes, activation='relu')
    
        def call(self, x, training=False):
            
            x = self.conv1_1(x)
            x = self.conv1_2(x)
            x = self.max_pool1(x)
    
            x = self.conv2_1(x)
            x = self.conv2_2(x)
            x = self.max_pool2(x)
    
            x = self.conv3_1(x)
            x = self.conv3_2(x)
            x = self.conv3_3(x)
            x = self.max_pool3(x)
    
            x = self.conv4_1(x)
            x = self.conv4_2(x)
            x = self.conv4_3(x)
            x = self.max_pool4(x)
    
            x = self.conv5_1(x)
            x = self.conv5_2(x)
            x = self.conv5_3(x)
            x = self.max_pool5(x)
    
            x = self.flat(x)
            x = self.dense1(x)
            x = self.dense2(x)
            x = self.dense3(x)
    
            return x
        
    
    
    model = VGG16(1000)
    model.build((1, 224, 224, 3))
    model.summary()