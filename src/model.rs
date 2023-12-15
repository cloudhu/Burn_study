use burn::{
    tensor::{
        Int,
        Tensor,
        backend::{
            Backend,
            AutodiffBackend
        }
    },
    nn::{
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
        loss::CrossEntropyLoss,
        Dropout,
        DropoutConfig,
        Linear,
        LinearConfig,
        ReLU,
        conv::{Conv2d, Conv2dConfig}
    },
    module::Module,
    config::Config,
    train::{
        ClassificationOutput,
        TrainOutput,
        TrainStep,
        ValidStep
    }
};
use crate::data::MNISTBatch;

// 我们的目标是创建一个用于图像分类的基本卷积神经网络。
// 我们将使用两个卷积层，然后是两个线性层，一些池化和ReLU激活来保持模型的简单性。
// 我们还将使用dropout来提高训练性能。
#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    //两个2d卷积神经层
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    //缓存池
    pool: AdaptiveAvgPool2d,
    //随机淘汰一部分神经元，从而提升性能
    dropout: Dropout,
    //两个线性层
    linear1: Linear<B>,
    linear2: Linear<B>,
    //激活函数采用修正线性单元ReLU
    activation: ReLU,
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    num_classes: usize,
    hidden_size: usize,
    #[config(default = "0.5")]
    dropout: f64,
}

impl ModelConfig {
    /// Returns the initialized model.
    pub fn init<B: Backend>(&self) -> Model<B> {
        Model {
            conv1: Conv2dConfig::new([1, 8], [3, 3]).init(),//kernel_size:在所有维度上使用3的内核大小
            conv2: Conv2dConfig::new([8, 16], [3, 3]).init(),
            //我们还使用自适应平均池化模块将图像的维数降为8 × 8矩阵的输出大小output_size
            pool: AdaptiveAvgPool2dConfig::new([8, 8]).init(),
            activation: ReLU::new(),
            //我们将在前向传递中将其平坦化，从而得到1024(16 _ 8 _ 8)张量。
            linear1: LinearConfig::new(16 * 8 * 8, self.hidden_size).init(),
            linear2: LinearConfig::new(self.hidden_size, self.num_classes).init(),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }

    /// Returns the initialized model using the recorded weights.
    /// 使用训练记录的模型数据来初始化模型，里面包含训练好的权重
    pub fn init_with<B: Backend>(&self, record: ModelRecord<B>) -> Model<B> {
        Model {
            conv1: Conv2dConfig::new([1, 8], [3, 3]).init_with(record.conv1),
            conv2: Conv2dConfig::new([8, 16], [3, 3]).init_with(record.conv2),
            pool: AdaptiveAvgPool2dConfig::new([8, 8]).init(),
            activation: ReLU::new(),
            linear1: LinearConfig::new(16 * 8 * 8, self.hidden_size).init_with(record.linear1),
            linear2: LinearConfig::new(self.hidden_size, self.num_classes)
                .init_with(record.linear2),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

impl<B: Backend> Model<B> {
    //前向传播
    /// # Shapes
    ///   - Images [batch_size, height, width]
    ///   - Output [batch_size, num_classes]
    pub fn forward(&self, images: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch_size, height, width] = images.dims();

        // Create a channel at the second dimension.
        let x = images.reshape([batch_size, 1, height, width]);

        let x = self.conv1.forward(x); // [batch_size, 8, _, _]
        let x = self.dropout.forward(x);
        let x = self.conv2.forward(x); // [batch_size, 16, _, _]
        let x = self.dropout.forward(x);
        let x = self.activation.forward(x);

        let x = self.pool.forward(x); // [batch_size, 16, 8, 8]
        let x = x.reshape([batch_size, 16 * 8 * 8]);
        let x = self.linear1.forward(x);
        let x = self.dropout.forward(x);
        let x = self.activation.forward(x);

        self.linear2.forward(x) // [batch_size, num_classes]
    }

    //前向传播分类:解决分类问题，这里就是将图片分类成对应的整型数字，也就是识别图像中的数字
    pub fn forward_classification(
        &self,
        images: Tensor<B, 3>,
        targets: Tensor<B, 1, Int>,
    ) -> ClassificationOutput<B> {
        let output = self.forward(images);//调用前向传播，拿到输出
        //通过交叉熵计算损失
        let loss = CrossEntropyLoss::new(None).forward(output.clone(), targets.clone());
        //分类输出
        ClassificationOutput::new(loss, output, targets)
    }
}

//实现训练步骤
impl<B: AutodiffBackend> TrainStep<MNISTBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, batch: MNISTBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(batch.images, batch.targets);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

//实现验证步骤
impl<B: Backend> ValidStep<MNISTBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, batch: MNISTBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch.images, batch.targets)
    }
}
