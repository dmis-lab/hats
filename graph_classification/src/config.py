import argparse

def get_args():
    argp = argparse.ArgumentParser(description='Stock Movement Prediction',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Direcotories
    argp.add_argument('--market_data_dir', type=str, default="./data/price/")
    argp.add_argument('--relation_data_dir', type=str, default="./data/relation/")
    argp.add_argument('--data_type', type=str, choices=['S5CONS', 'S5ENRS', 'S5UTIL', 'S5FINL', 'S5INFT'], required=True)
    argp.add_argument('--save_dir',  type=str, default="./out")
    argp.add_argument('--model_dir', type=str, default="./model")
    argp.add_argument('--model_id', type=str, default="epoch200.pt")
    argp.add_argument('--save_log', action='store_true', default=False)
    argp.add_argument('--max_to_keep', type=int, default=10)
    argp.add_argument('--pretrained_dir', type=str, default=None)

    # Data Control
    argp.add_argument('--preprocess', action='store_true', default=False)
    argp.add_argument('--dev_size', type=int, default=50)
    argp.add_argument('--test_size', type=int, default=100)
    argp.add_argument('--test_phase', type=int)
    argp.add_argument('--train_proportion', type=float, default=3.0)
    argp.add_argument('--feature_list', nargs='+', type=str, help='return, volume, close')
    argp.add_argument('--num_relations', type=int, default=0)
    argp.add_argument('--use_rel_list', type=list,
                    default=[2,3,6,11,13,17,18,28,30,32,37,38,39,40,41,48,57,84,85,86])
    argp.add_argument('--max_relations', type=int, default=5)
    argp.add_argument('--min_train_period', type=int, default=300)
    argp.add_argument('--label_proportion', nargs='+', type=int, required=True)
    argp.add_argument('--stack_layer', type=int, default=1)
    argp.add_argument('--node_features', type=int, default=64)
    argp.add_argument('--use_bias', action='store_true', default=True)
    argp.add_argument('--adj_train_phase', type=str, default='adj_sum', choices=['adj_sum', 'zero_padding'])
    argp.add_argument('--assignment_layer', type=list, default=[10, 1])
    argp.add_argument('--pred_threshold', type=float, default=0.6)
    argp.add_argument('--num_layer', type=int, default=1)

    # Main Control
    argp.add_argument('--train_on_stock', action='store_true', default=False)
    argp.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    argp.add_argument('--lookback', type=int, default=50)
    argp.add_argument('--batch_size', type=int, default=32)
    argp.add_argument('--n_epochs', type=int, default=200)
    argp.add_argument('--n_iter_per_epoch', type=int, default=50)
    argp.add_argument('--eval_step', type=int, default=10)
    argp.add_argument('--print_step', type=int, default=10)
    argp.add_argument('--early_stop_type', type=str, default='acc', choices=['acc', 'loss', 'f1', 'mcc'])

    ## model general
    argp.add_argument('--model_type', type=str, default='graph-HATS',
            choices=['mlp', 'lstm', 'gru', 'cnn', 'graph-cnn-pooling', 'graph-HATS'])
    argp.add_argument('--price_model', type=str, default='LSTM', choices=['CNN', 'LSTM', 'GRU'])
    argp.add_argument('--GNN_model', type=str, default='HATS', choices=['GCN', 'TGC', 'HATS'])
    argp.add_argument('--feat_att', action='store_true', default=False)
    argp.add_argument('--rel_att', action='store_true', default=False)
    argp.add_argument('--inference_model', type=str, default='CNN', choices=['CNN', 'LSTM', 'GRU'])

    ## optimizer
    argp.add_argument('--optimizer', type=str, default='Adam')
    argp.add_argument('--lr', type=float, default=5e-5)
    argp.add_argument('--weight_decay', type=float, default=1e-5)
    argp.add_argument('--dropout', type=float, default=0.3)
    argp.add_argument('--momentum', type=float, default=0.9) #optimizer
    argp.add_argument('--grad-max-norm', type=float, default=2.0)

    return argp.parse_args()
