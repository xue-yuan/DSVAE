import torch

from models import ShareEncoder, SpeakerEncoder, ContentEncoder
from models import PreNetBlock, PreNet
from models import PostNetBlock, PostNet
from models import Encoder, Decoder
from models import DSVAE
from teacherstudent.models import StudentDSVAE1, StudentDSVAE2


def test_model(model, x):
    output = model(x)
    if isinstance(output, tuple):
        outputs = []
        for o in output:
            outputs.append(o.shape)

        return outputs
    else:
        return output.shape


batch_size = 2
input_channel = 1
input_height = 32
input_width = 4096


def test_share_encoder():
    o = test_model(
        ShareEncoder(),
        torch.randn(batch_size, input_channel, input_height, input_width),
    )

    print(o)

    assert o[0] == batch_size
    assert o[1] == 256
    assert o[2] == input_height
    assert o[3] == input_width


def test_speaker_encoder():
    z = torch.randn(batch_size, 256, input_height, input_width)
    z = z.permute(0, 3, 1, 2)
    z = z.reshape(z.shape[0], z.shape[1], -1)
    o = test_model(
        SpeakerEncoder(),
        z,
    )

    print(o)

    assert o[0][0] == batch_size
    assert o[0][1] == 64
    assert o[1][0] == batch_size
    assert o[1][1] == 64


def test_content_encoder():
    z = torch.randn(batch_size, 256, input_height, input_width)
    z = z.permute(0, 3, 1, 2)
    z = z.reshape(z.shape[0], z.shape[1], -1)
    o = test_model(
        ContentEncoder(),
        z,
    )

    print(o)

    assert o[0][0] == batch_size
    assert o[0][1] == 64
    assert o[1][0] == batch_size
    assert o[1][1] == 64


def test_encoder():
    o = test_model(
        Encoder(),
        torch.randn(batch_size, input_channel, input_height, input_width),
    )

    print(o)

    assert o[0][0] == batch_size
    assert o[0][1] == 64
    assert o[1][0] == batch_size
    assert o[1][1] == 64
    assert o[2][0] == batch_size
    assert o[2][1] == 64
    assert o[3][0] == batch_size
    assert o[3][1] == 64


def test_pre_net_block():
    o = test_model(
        PreNetBlock(),
        torch.randn(batch_size, input_channel, 128),
    )

    print(o)

    assert o[0] == batch_size
    assert o[1] == 512
    assert o[2] == 128


def test_pre_net():
    o = test_model(
        PreNet(),
        torch.randn(batch_size, input_channel, 128),
    )

    print(o)

    assert o[0] == batch_size
    assert o[1] == 512
    assert o[2] == 128


def test_post_net_block():
    o = test_model(
        PostNetBlock(),
        torch.randn(batch_size, 512, 128),
    )

    print(o)

    assert o[0] == batch_size
    assert o[1] == 512
    assert o[2] == 128


def test_post_net():
    o = test_model(
        PostNet(),
        torch.randn(batch_size, 512, 128),
    )

    print(o)

    assert o[0] == batch_size
    assert o[1] == 512
    assert o[2] == 128


def test_decoder():
    o = test_model(
        Decoder(),
        torch.randn(batch_size, input_channel, 128),
    )

    print(o)

    assert o[0] == batch_size
    assert o[1] == input_channel
    assert o[2] == input_height
    assert o[3] == input_width


def test_dsvae():
    o = test_model(
        DSVAE(),
        torch.randn(batch_size, input_channel, input_height, input_width),
    )

    print(o)

    assert o[0][0] == batch_size
    assert o[0][1] == input_channel
    assert o[0][2] == input_height
    assert o[0][3] == input_width

    assert o[1][0] == batch_size
    assert o[1][1] == 64
    assert o[2][0] == batch_size
    assert o[2][1] == 64
    assert o[3][0] == batch_size
    assert o[3][1] == 64
    assert o[4][0] == batch_size
    assert o[4][1] == 64


def test_student_dsvae1():
    o = test_model(
        StudentDSVAE1(),
        torch.randn(batch_size, input_channel, input_height, input_width),
    )

    print(o)

    assert o[0][0] == batch_size
    assert o[0][1] == input_channel
    assert o[0][2] == input_height
    assert o[0][3] == input_width

    assert o[1][0] == batch_size
    assert o[1][1] == 64
    assert o[2][0] == batch_size
    assert o[2][1] == 64
    assert o[3][0] == batch_size
    assert o[3][1] == 64
    assert o[4][0] == batch_size
    assert o[4][1] == 64


def test_student_dsvae2():
    o = test_model(
        StudentDSVAE2(),
        torch.randn(batch_size, input_channel, input_height, input_width),
    )

    print(o)

    assert o[0][0] == batch_size
    assert o[0][1] == input_channel
    assert o[0][2] == input_height
    assert o[0][3] == input_width

    assert o[1][0] == batch_size
    assert o[1][1] == 64
    assert o[2][0] == batch_size
    assert o[2][1] == 64
    assert o[3][0] == batch_size
    assert o[3][1] == 64
    assert o[4][0] == batch_size
    assert o[4][1] == 64


# test_share_encoder()
# test_speaker_encoder()
# test_content_encoder()
# test_encoder()
# test_pre_net_block()
# test_pre_net()
# test_post_net_block()
# test_post_net()
# test_decoder()
# test_dsvae()
test_student_dsvae1()
# test_student_dsvae2()
