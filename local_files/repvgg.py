#1. add softmax when export onnx model


        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        if torch.onnx.is_in_onnx_export():
            return torch.nn.functional.softmax(out, dim=1)
        return out


optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26] 
