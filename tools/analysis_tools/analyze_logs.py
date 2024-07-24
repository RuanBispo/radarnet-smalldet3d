# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
import numpy as np
import seaborn as sns
from collections import defaultdict
from matplotlib import pyplot as plt
import glob


def cal_train_time(log_dicts, args):
    for i, log_dict in enumerate(log_dicts):
        print(f'{"-" * 5}Analyze train time of {args.json_logs[i]}{"-" * 5}')
        # all_times = []
        all_times = np.array([])
        for epoch in log_dict.keys():
            if args.include_outliers:
                all_times = all_times.append(log_dict[epoch]['time'])
            else:
                all_times.append(log_dict[epoch]['time'][1:])
        # all_times = np.array(all_times, dtype=object)
        epoch_ave_time = all_times.mean(-1)
        slowest_epoch = epoch_ave_time.argmax()
        fastest_epoch = epoch_ave_time.argmin()
        std_over_epoch = epoch_ave_time.std()
        print(f'slowest epoch {slowest_epoch + 1}, '
              f'average time is {epoch_ave_time[slowest_epoch]:.4f}')
        print(f'fastest epoch {fastest_epoch + 1}, '
              f'average time is {epoch_ave_time[fastest_epoch]:.4f}')
        print(f'time std over epochs is {std_over_epoch:.4f}')
        print(f'average iter time: {np.mean(all_times):.4f} s/iter')
        print()


def plot_curve(log_dicts, args):
    # plt.style.use('dark_background')
    if args.backend is not None:
        plt.switch_backend(args.backend)
    sns.set_style(args.style)
    # if legend is None, use {filename}_{key} as legend
    legend = args.legend
    if legend is None:
        legend = []
        
        
        for json_log in args.json_logs:
            for metric in args.keys:
                legend.append(f'{metric}')
    assert len(legend) == (len(args.json_logs) * len(args.keys))
    metrics = args.keys

    num_metrics = len(metrics)
    print('===========',args.out[-19:-4], '(5 epochs)','===========')
    for i, log_dict in enumerate(log_dicts):
        epochs = list(log_dict.keys())
        for j, metric in enumerate(metrics):
            # print(f'plot curve of {args.json_logs[i]}, metric is {metric}')
            if metric not in log_dict[epochs[args.interval - 1]]:
                raise KeyError(
                    f'{args.json_logs[i]} does not contain metric {metric}')

            if args.mode == 'eval':
                if min(epochs) == args.interval:
                    x0 = args.interval
                else:
                    # if current training is resumed from previous checkpoint
                    # we lost information in early epochs
                    # `xs` should start according to `min(epochs)`
                    if min(epochs) % args.interval == 0:
                        x0 = min(epochs)
                    else:
                        # find the first epoch that do eval
                        x0 = min(epochs) + args.interval - \
                            min(epochs) % args.interval
                xs = np.arange(x0, max(epochs) + 1, args.interval)
                ys = []
                for epoch in epochs[args.interval - 1::args.interval]:
                    ys += log_dict[epoch][metric]

                # if training is aborted before eval of the last epoch
                # `xs` and `ys` will have different length and cause an error
                # check if `ys[-1]` is empty here
                if not log_dict[epoch][metric]:
                    xs = xs[:-1]

                ax = plt.gca()
                ax.set_xticks(xs)
                plt.xlabel('epoch')
                plt.plot(xs, ys, label=legend[i * num_metrics + j], marker='o')
            else:
                xs = []
                ys = []
                num_iters_per_epoch = \
                    log_dict[epochs[args.interval-1]]['iter'][-1]
                for epoch in epochs[args.interval - 1::args.interval]:
                    iters = log_dict[epoch]['iter']
                    if log_dict[epoch]['mode'][-1] == 'val':
                        iters = iters[:-1]
                    xs.append(
                        np.array(iters) + (epoch - 1) * num_iters_per_epoch)
                    ys.append(np.array(log_dict[epoch][metric][:len(iters)]))
                xs = np.concatenate(xs)
                ys = np.concatenate(ys)
                # print(xs.shape)
                # ax.set_yticks(np.arange(0, 1., 0.7))
                plt.xlabel('iter')
                plt.plot(xs, 
                         ys, 
                         label=legend[i * num_metrics + j], 
                         linewidth=0.5, 
                         alpha=0.8)
                #calculate equation for quadratic trendline
                # offset = int(len(xs)/4)
                offset = 3 #5
                z = np.polyfit(np.log(xs[offset:]), ys[offset:], 1)
                # z = np.polyfit(xs[offset:], ys[offset:], 1)
                p = np.poly1d(z)
                y = p(np.log(xs[offset:]))
                # y = p(xs[offset:])
                plt.plot(xs[offset:], 
                         y, 
                         label=f'{np.mean(ys[-10:]).round(4)}', 
                         linewidth=0.4, 
                         color='black',
                         alpha=0.3)
                # pred = np.round(p(np.log(28000))*1.05,4)
                pred_2_epochs = np.round(p(np.log(24*28000)),4)
                # pred_2_epochs = np.round(p(28000),4)
                
                plt.plot(xs[-1],
                         pred_2_epochs,
                         color='red',
                         marker='.',
                         linewidth=0.1, 
                         )
                plt.annotate(f' ≈{pred_2_epochs}',
                xy=(xs[-1], pred_2_epochs), xycoords='data',
                # xytext=(xs[-1], pred), textcoords='offset points',
                # arrowprops=dict(facecolor='black', shrink=0.05),
                horizontalalignment='left', verticalalignment='top')
                
                print(f'{metric} = {pred_2_epochs} (≈{np.mean(ys[-10:]).round(4)})', end =" ")
            # Shrink current axis's height by 10% on the bottom
            # plt.ylim(0, 5)
            ax = plt.gca()
            ax.set_yticks(np.arange(0, 0.61, 0.05))
        
            # ax.grid(which = "major", linewidth = 1, color = "white")
            # ax.grid(which = "minor", linewidth = 0.5, color = "white")
            # ax.minorticks_on()
            # ax.xaxis.set_tick_params(which='minor', bottom=False)

            # Put a legend below current axis
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.10),
                    fancybox=True, ncol=3)
            plt.ylim(-0.01, 0.61)
            # plt.legend()
        if args.title is not None:
            plt.title(f'{args.title}', pad=27)
        
    
    plt.grid()
    
    if args.out is None:
        plt.show()
    else:
        print()
        print(f'save curve to: {args.out}')
        print()
        plt.savefig(args.out)
        plt.cla()


def add_plot_parser(subparsers):
    parser_plt = subparsers.add_parser(
        'plot_curve', help='parser for plotting curves')
    parser_plt.add_argument(
        'json_logs',
        type=str,
        nargs='+',
        help='path of train log in json format')
    parser_plt.add_argument(
        '--keys',
        type=str,
        nargs='+',
        default=['mAP_0.25'],
        help='the metric that you want to plot')
    parser_plt.add_argument('--title', type=str, help='title of figure')
    parser_plt.add_argument(
        '--legend',
        type=str,
        nargs='+',
        default=None,
        help='legend of each plot')
    parser_plt.add_argument(
        '--backend', type=str, default=None, help='backend of plt')
    parser_plt.add_argument(
        '--style', type=str, default='dark', help='style of plt')
    parser_plt.add_argument('--out', type=str, default=None)
    parser_plt.add_argument('--mode', type=str, default='train')
    parser_plt.add_argument('--interval', type=int, default=1)


def add_time_parser(subparsers):
    parser_time = subparsers.add_parser(
        'cal_train_time',
        help='parser for computing the average time per training iteration')
    parser_time.add_argument(
        'json_logs',
        type=str,
        nargs='+',
        help='path of train log in json format')
    parser_time.add_argument(
        '--include-outliers',
        action='store_true',
        help='include the first value of every epoch when computing '
        'the average time')


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze Json Log')
    # currently only support plot curve and calculate average train time
    subparsers = parser.add_subparsers(dest='task', help='task parser')
    add_plot_parser(subparsers)
    add_time_parser(subparsers)
    args = parser.parse_args()
    return args


def load_json_logs(json_logs):
    # load and convert json_logs to log_dict, key is epoch, value is a sub dict
    # keys of sub dict is different metrics, e.g. memory, bbox_mAP
    # value of sub dict is a list of corresponding values of all iterations
    log_dicts = [dict() for _ in json_logs]
    for json_log, log_dict in zip(json_logs, log_dicts):
        with open(json_log, 'r') as log_file:
            for line in log_file:
                log = json.loads(line.strip())
                # skip lines without `epoch` field
                if 'epoch' not in log:
                    continue
                epoch = log.pop('epoch')
                if epoch not in log_dict:
                    log_dict[epoch] = defaultdict(list)
                for k, v in log.items():
                    log_dict[epoch][k].append(v)
    return log_dicts


def main():
    args = parse_args()

    json_logs = args.json_logs
    
    # print(json_logs)
    
    # json_logs = glob.glob(f"{json_logs[0]}/*.json")
    
    # if json_logs:
    #     json_logs = args.json_logs
    
    for json_log in json_logs:
        # print(json_log)
        assert json_log.endswith('.json')
    # print(json_logs)

    log_dicts = load_json_logs(json_logs)

    eval(args.task)(log_dicts, args)


if __name__ == '__main__':
    main()