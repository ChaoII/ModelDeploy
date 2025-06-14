from modeldeploy.vision import LetterBoxRecord
import json

record_ = LetterBoxRecord(
    ipt_h=1.0,
    ipt_w=1.0,
    out_h=1.0,
    out_w=1.0,
    pad_h=1.0,
    pad_w=1.0,
    scale=1.0,
)


def letterbox_record_to_json(record: LetterBoxRecord):
    print(record.__dict__)


letterbox_record_to_json(record_)
