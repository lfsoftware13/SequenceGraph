import time
import os


import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import typing
from toolz.sandbox import unzip
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.metrics import log_loss
import numpy as np
import more_itertools


from common import torch_util, util, problem_util
from common.evaluate_util import Evaluator
from common.problem_util import to_cuda
from common.util import data_loader
from read_data.data_set import OriDataSet


def get_model(model_fn, parameter, pre_process_module_fn, pre_process_module_parameter, path, load_previous=False,):
    m = model_fn(
        **parameter
    )
    # to_cuda(m)
    m = nn.DataParallel(m.cuda(), device_ids=[0, 1])
    m = pre_process_module_fn(m, **pre_process_module_parameter)
    if load_previous:
        torch_util.load_model(m, path)
        print("load previous model")
    else:
        print("create new model")
    return m


def train(model, dataset, batch_size, loss_function, optimizer, clip_norm, epoch_ratio):
    total_loss = to_cuda(torch.Tensor([0]))
    steps = to_cuda(torch.Tensor([0]))
    # previous_char_max = 0
    # previous_word_max = 0
    model.train()
    for batch_data in data_loader(dataset, batch_size=batch_size, is_shuffle=True, drop_last=True, epoch_ratio=epoch_ratio):
        # print(batch_data['terminal_mask'])
        # print('batch_data size: ', len(batch_data['terminal_mask'][0]), len(batch_data['terminal_mask'][0][0]))
        # res = list(more_itertools.collapse(batch_data['terminal_mask']))
        # print('res len: ', len(res))
        # res = util.padded(batch_data['terminal_mask'], deepcopy=True, fill_value=0)
        # print('batch_data size: ', len(res[0]), len(res[0][0]))
        # res = list(more_itertools.collapse(res))
        # print('res len: ', len(res))
        # previous_char_max = max(previous_char_max, max(batch_data['q1_char_length']), max(batch_data['q2_char_length']))
        # previous_word_max = max(previous_word_max, max(batch_data['q1_word_length']), max(batch_data['q2_word_length']))
        # print('max q1_length:{},{}'.format(max(batch_data['q1_char_length']), max(batch_data['q1_word_length'])))
        # print("max q2_length:{},{}".format(max(batch_data['q2_char_length']), max(batch_data['q2_word_length'])))
        # print("previous_char_max:{}, previous_word_max:{}".format(previous_char_max, previous_word_max))
        model.zero_grad()
        log_probs = model.forward(
            batch_data
        )
        # log_probs.register_hook(create_hook_fn("log_probs"))

        # print("log_probs sizze:{}".format(log_probs.size()))
        label = to_cuda(torch.LongTensor(batch_data['label']))
        loss = loss_function(log_probs, label)

        # loss.register_hook(create_hook_fn("loss"))
        loss.backward()

        if clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)

        # print()
        # print("The loss is nan:{}".format(is_nan(loss.detach())))
        # print("The loss grad is nan:{}".format(is_nan(loss.grad)))
        # print("The log_probs is nan:{}".format(is_nan(log_probs.detach())))
        # print("The log_probs grad is nan:{}".format(is_nan(log_probs.grad)))
        # for name, param in model.named_parameters():
        #     print("name of {}: has nan:{}".format(name, is_nan(param.detach())))
        #     print("the gradient of {}: has nan:{}".format(name, is_nan(param.grad)))
        # if HAS_NAN:
        #     for k, v in batch_data.items():
        #         print("{}:{}".format(k, show_tensor(v)))
        #     print("{}:{}".format("target", show_tensor(target)))
        # print()

        optimizer.step()

        # print("loss：{}".format(loss.data))
        total_loss += loss.data
        steps += 1
    return total_loss/steps


def evaluate(model, valid_dataset, batch_size, evaluate_object_list: typing.List[Evaluator], train_loss_function):
    model.eval()
    for o in evaluate_object_list:
        o.clear_result()
    train_total_loss = to_cuda(torch.Tensor([0]))
    steps = to_cuda(torch.Tensor([0]))
    for batch_data in data_loader(valid_dataset, batch_size=batch_size, is_shuffle=False, drop_last=False):
        model.zero_grad()
        predict_logit = model.forward(batch_data)
        target = to_cuda(torch.LongTensor(batch_data['label']))
        train_loss = train_loss_function(predict_logit, target)
        for evaluator in evaluate_object_list:
            evaluator.add_result(predict_logit, target)
        train_total_loss += train_loss.data
        steps += 1
    return evaluate_object_list, train_total_loss/steps


def train_and_evaluate(
        model, batch_size,
        train_dataset, valid_dataset, test_dataset,
        epoches,
        lr,
        load_previous,
        save_path,
        train_loss_function,
        clip_norm,
        optimizer,
        optimizer_dict,
        just_evaluate,
        epoch_ratio,
        evaluate_object_list,
        ):
    optimizer = optimizer(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, **optimizer_dict)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    if load_previous:
        valid_loss, train_valid_loss = evaluate(model, valid_dataset, batch_size, evaluate_object_list,
                                                train_loss_function)
        # test_loss, train_test_loss = evaluate(model, test_dataset, batch_size, evaluate_loss_function,
        #                                       train_loss_function)
        best_valid_loss = valid_loss
        # best_test_loss = test_loss
        print("load the previous mode")
        print("train validation loss is:{}".format(train_valid_loss,))
        for evaluator in valid_loss:
            print(evaluator)
        scheduler.step(best_valid_loss)
        if just_evaluate:
            print("just evaluate return")
            return
    else:
        best_valid_loss = None
        # best_test_loss = None

    begin_time = time.time()
    # with torch.autograd.profiler.profile() as prof:
    for epoch in range(epoches):
        train_loss = train(model, train_dataset, batch_size, train_loss_function, optimizer, clip_norm, epoch_ratio)
        valid_loss, train_valid_loss = evaluate(model, valid_dataset, batch_size, evaluate_object_list,
                                                train_loss_function)
        # test_loss, train_test_loss = evaluate(model, test_dataset, batch_size, evaluate_loss_function,
        #                                       train_loss_function)

        train_loss = train_loss.item()
        train_valid_loss = train_valid_loss.item()
        # train_test_loss = train_test_loss.item()

        scheduler.step(train_valid_loss)

        if best_valid_loss is None or train_valid_loss > best_valid_loss:
            best_valid_loss = train_valid_loss
            # best_test_loss = train_test_loss
            torch_util.save_model(model, save_path)

        print("epoch {}".format(epoch))
        print("train loss of {},  train validation loss is:{}".format(train_loss, train_valid_loss, ))
        for evaluator in valid_loss:
            print(evaluator)
    # print(prof)
    print("The model {} best valid loss is {}".
          format(save_path, best_valid_loss))
    print("use time {} seconds".format(time.time() - begin_time))


if __name__ == '__main__':
    import parameter_config
    import config
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--load_previous", type=bool, default=False)
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--config_name", type=str)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--parallel", type=bool)
    parser.add_argument("--just_evaluate", type=bool, default=False)
    args = parser.parse_args()
    load_previous = args.load_previous
    problem_util.GPU_INDEX = args.gpu
    problem_util.Parallel = args.parallel
    is_debug = args.debug
    just_evaluate = args.just_evaluate

    p_config = parameter_config.__dict__.get(args.config_name)(is_debug)
    epoches = p_config.get("epcohes", 20)
    lr = p_config.get("lr", 20)
    batch_size = p_config.get("batch_size", 32)
    train_loss_fn = p_config.get("train_loss", nn.CrossEntropyLoss)()
    clip_norm = p_config.get("clip_norm", 10)
    optimizer = p_config.get("optimizer", optim.SGD)
    optimizer_dict = p_config.get("optimizer_dict", dict())
    epoch_ratio = p_config.get("epoch_ratio", 0.5)
    evaluate_object_list = p_config.get("evaluate_object_list")
    save_root_path = os.path.join(config.DATA_PATH, p_config.get("name"))
    util.make_dir(save_root_path)
    need_pad = p_config.get("need_pad", False)
    print("save_root_path:{}".format(save_root_path))
    model_path = os.path.join(save_root_path, "model.pkl")
    model = get_model(
        p_config['model_fn'],
        p_config['model_dict'],
        p_config['pre_process_module_fn'],
        p_config['pre_process_module_dict'],
        model_path,
        load_previous=load_previous,
    )

    train_data, val_data, test_data = p_config.get("data")
    print("The size of train data: {}".format(len(train_data)))
    print("The size of val data: {}".format(len(val_data)))
    print("The size of test data: {}".format(len(test_data)))
    train_and_evaluate(model, batch_size, train_data, val_data, test_data, epoches, lr, load_previous, model_path,
                       train_loss_fn, clip_norm, optimizer, optimizer_dict, just_evaluate, epoch_ratio,
                       evaluate_object_list)

    test_loss, train_test_loss = evaluate(model, test_data, batch_size, evaluate_object_list,
                                          train_loss_fn)
    print("train_test_loss is {}".format(train_test_loss.item(),))
    for o in  test_loss:
        print(o)


