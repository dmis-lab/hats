import argparse

def get_args():
    argp = argparse.ArgumentParser(description='Stock Movement Prediction',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Direcotories
    argp.add_argument('--market_data_dir', type=str, default="./data/price")
    argp.add_argument('--relation_data_dir', type=str, default="./data/relation")
    argp.add_argument('--data_type', type=str, default='snp500')
    argp.add_argument('--save_dir',  type=str, default="./out")
    argp.add_argument('--model_dir', type=str, default="./model")
    argp.add_argument('--model_id', type=str, default="epoch300.pt")
    argp.add_argument('--save_log', action='store_true', default=False)
    argp.add_argument('--max_to_keep', type=int, default=10)
    argp.add_argument('--load', type=str, default='./out')

    # Data Control
    argp.add_argument('--dev_size', type=int, default=50)
    argp.add_argument('--test_size', type=int, default=100)
    argp.add_argument('--test_phase', type=int)
    argp.add_argument('--train_proportion', type=float, default=3.0)
    argp.add_argument('--feature_list', nargs='+', type=str, help='return, volume, close, technical')
    argp.add_argument('--num_relations', type=int, default=0)
    argp.add_argument('--num_companies', type=int, default=0)
    no_use = [0,1,2,6,13,20,22,29,31,32,44,45,46,47]
    argp.add_argument('--use_rel_list', nargs='+', type=int,
                    default=[i for i in range(85) if i not in no_use])
    argp.add_argument('--neighbors_sample', type=int, default=20)
    argp.add_argument('--att_topk', type=int, default=50)
    argp.add_argument('--min_train_period', type=int, default=300)
    argp.add_argument('--label_proportion', nargs='+', type=int, default=[1, 1, 1])

    # Main Control
    argp.add_argument('--train_on_stock', action='store_true', default=False)
    argp.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    argp.add_argument('--lookback', type=int, default=50)
    argp.add_argument('--batch_size', type=int, default=32)
    argp.add_argument('--n_epochs', type=int, default=300)
    argp.add_argument('--eval_step', type=int, default=1)
    argp.add_argument('--print_step', type=int, default=10)
    argp.add_argument('--early_stop_type', type=str, default='acc', choices=['acc', 'loss', 'f1', 'mcc'])

    ## model general
    argp.add_argument('--model_type', type=str, default='HATS')
    argp.add_argument('--node_feat_size', type=int, default=128)
    argp.add_argument('--rel_projection', action='store_true', default=False)
    argp.add_argument('--feat_att', action='store_true', default=False)
    argp.add_argument('--rel_att', action='store_true', default=False)
    argp.add_argument('--num_layer', type=int, default=1)
    argp.add_argument('--inference_model', type=str, default='lstm')

    ## optimizer
    argp.add_argument('--optimizer', type=str, default='Adam')
    argp.add_argument('--lr', type=float, default=5e-4)
    argp.add_argument('--weight_decay', type=float, default=5e-5)
    argp.add_argument('--dropout', type=float, default=0.5)
    argp.add_argument('--momentum', type=float, default=0.9) #optimizer
    argp.add_argument('--grad-max-norm', type=float, default=2.0)

    return argp.parse_args()
