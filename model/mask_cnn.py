from torch import nn


class MaskConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super(MaskConv2d, self).__init__()
        self.conv2d = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding,
                                bias=False, stride=1)

    def forward(self, x, mask):
        """

        :param x: bsz x dim x max_len x max_len
        :param mask: bsz x 1 x max_len x max_len
        :return:
        """
        x = x.masked_fill(mask, 0)
        _x = self.conv2d(x)
        return _x
