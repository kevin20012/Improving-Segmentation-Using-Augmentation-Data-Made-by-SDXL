import os
import argparse
from ast import literal_eval

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_log_path', type=str, help="json 로그 파일의 위치")
    # parser.add_argument('--metric', default='mIoU',type=str, help="평가방법 선택 mIoU, aAcc, mAcc")

    return parser.parse_args()

def main():
    args = parser_args()
    json_log = args.json_log_path
    # metric = args.metric

    f = open(json_log, 'r')
    text = f.read()
    text = text.split("\n")

    best_step = 0
    best_score = 0
    for line in text:
        try:
            dic = literal_eval(line)
            if 'mIoU' not in dic.keys():
                pass
            else:
                cur_value = (dic['mAcc']+dic['mIoU'])/3 #모든 metric을 고려
                if best_score <= cur_value:
                    best_step = dic['step']
                    best_score = cur_value
        except:
            pass

    f.close()
    
    print(f"{best_step}")

if __name__ == "__main__":
    main()