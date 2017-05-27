class block(nn.Module):
    def __init__(self, ci, co, stride, p=0.0):
        super(block, self).__init__()
        self.bn1 = nn.BatchNorm2d(ci)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(ci, co, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(co)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(co, co, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.p = p
        self.eq = (ci == co)
        self.convShortcut = (not self.eq) and nn.Conv2d(ci, co, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
    def forward(self, x):
        if not self.eq:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(self.eq and out or x)))
        if self.p > 0:
            out = F.dropout(out, p=self.p, training=self.training)
        out = self.conv2(out)
        return th.add((not self.eq) and self.convShortcut(x) or x, out)

class netblock(nn.Module):
    def __init__(self, nl, ci, co, block, stride, p=0.0):
        super(netblock, self).__init__()
        self.layer = self._make_layer(block, ci, co, nl, stride, p)

    def _make_layer(self, block, ci, co, nl, stride, p):
        layers = []
        for i in range(nl):
            layers.append(block(i == 0 and ci or co, co, i == 0 and stride or 1, p))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

class wideresnet(nn.Module):
    def __init__(self, opt = {'d':0., 'depth':28, 'widen':10}):
        super(wideresnet, self).__init__()
        self.name = 'wideresnet'

        opt['d'] = 0.
        opt['depth'] = 40
        opt['widen'] = 4
        opt['l2'] = 5e-4
        p, depth, widen = opt['d'], opt['depth'], opt['widen']

        if opt['dataset'] == 'cifar10':
            num_classes = 10
        elif opt['dataset'] == 'cifar100':
            num_classes = 100

        nc = [16, 16*widen, 32*widen, 64*widen]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6

        self.conv1 = nn.Conv2d(3, nc[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.block1 = netblock(n, nc[0], nc[1], block, 1, p)
        self.block2 = netblock(n, nc[1], nc[2], block, 2, p)
        self.block3 = netblock(n, nc[2], nc[3], block, 2, p)
        self.bn1 = nn.BatchNorm2d(nc[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nc[3], num_classes)
        self.nc = nc[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nc)
        return self.fc(out)