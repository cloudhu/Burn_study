use burn::{
    optim::AdamConfig,
    backend::{Autodiff, Wgpu, wgpu::AutoGraphicsApi},
    config::Config,
    data::{
        dataset::source::huggingface::MNISTItem,
        dataloader::batcher::Batcher
    },
    module::Module,
    record::{CompactRecorder, Recorder},
    tensor::backend::Backend,
};
use my_burn_app::{
    model::ModelConfig,
    train::{train, TrainingConfig},
    data::MNISTBatcher
};

//用简单的2d卷积神经网络来识别0-9的数字图形
fn main() {
    //使用Wgpu作为后端来进行图形处理
    type MyBackend = Wgpu<AutoGraphicsApi, f32, i32>;

    //通过Autodiff结构体的包装来实现自动微分，消除不同后端带来的差异性
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = burn::backend::wgpu::WgpuDevice::default();
    //配置模型：类的数量10（0-9），隐藏层维度512，优化默认Adam，还有默认的后端设备；开始训练
    train::<MyAutodiffBackend>(
        "/tmp/guide",
        TrainingConfig::new(ModelConfig::new(10, 512), AdamConfig::new()),
        device,
    );

}

//使用训练好的数据进行推理
pub fn infer<B: Backend>(artifact_dir: &str, device: B::Device, item: MNISTItem) {
    //加载模型配置和训练结果记录器，并将模型加载到设备上
    let config = TrainingConfig::load(format!("{artifact_dir}/config.json"))
        .expect("Config should exist for the model");
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into())
        .expect("Trained model should exist");

    let model = config.model.init_with::<B>(record).to_device(&device);

    let label = item.label;
    let batcher = MNISTBatcher::new(device);
    let batch = batcher.batch(vec![item]);
    let output = model.forward(batch.images);
    let predicted = output.argmax(1).flatten::<1>(0, 1).into_scalar();

    println!("Predicted {} Expected {}", predicted, label);
}