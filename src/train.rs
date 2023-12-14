use burn::{
    config::Config,
    data::{
        dataloader::DataLoaderBuilder,
        dataset::source::huggingface::MNISTDataset
    },
    module::Module,
    optim::AdamConfig,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
    train::{
        LearnerBuilder,
        metric::{AccuracyMetric, LossMetric}
    }
};
use crate::data::MNISTBatcher;
use crate::model::ModelConfig;

//训练配置结构体
#[derive(Config)]
pub struct TrainingConfig {
    pub model: ModelConfig,//模型配置
    pub optimizer: AdamConfig,//优化器
    #[config(default = 10)]
    pub num_epochs: usize,//轮次
    #[config(default = 64)]
    pub batch_size: usize,//批处理大小
    #[config(default = 4)]
    pub num_workers: usize,//工作数量
    #[config(default = 42)]
    pub seed: u64,//随机种子
    #[config(default = 1.0e-4)]
    pub learning_rate: f64,//学习率
}

pub fn train<B: AutodiffBackend>(artifact_dir: &str, config: TrainingConfig, device: B::Device) {
    //首先确保artifact_dir参数路径可用
    std::fs::create_dir_all(artifact_dir).ok();
    //将训练配置保存到artifact_dir中
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");
    //配置随机种子
    B::seed(config.seed);
    //然后使用前面创建的批处理程序初始化数据加载器。
    let batcher_train = MNISTBatcher::<B>::new(device.clone());
    let batcher_valid = MNISTBatcher::<B::InnerBackend>::new(device.clone());

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(MNISTDataset::train());

    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(MNISTDataset::test());
    //根据配置构建学习机器
    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device])
        .num_epochs(config.num_epochs)
        .build(
            config.model.init::<B>(),
            config.optimizer.init(),
            config.learning_rate,
        );

    let model_trained = learner.fit(dataloader_train, dataloader_test);

    model_trained
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");
}