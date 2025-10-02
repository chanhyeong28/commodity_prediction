from data_provider.dataloader import Dataset_Mitsui_Commodity
from torch.utils.data import DataLoader

data_dict = {
    'commodity': Dataset_Mitsui_Commodity,
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    # Time-LlaMA: we don't use time encodings in Mitsui pipeline
    timeenc = 0
    percent = getattr(args, 'percent', 100)

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    # Mitsui commodity default path and parameters
    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        percent=percent,
        seasonal_patterns=getattr(args, 'seasonal_patterns', None),
        few_shot_ratio=getattr(args, 'few_shot_ratio', 1.0),
    )
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
