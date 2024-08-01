from __future__ import print_function, absolute_import, division
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.utils import shuffle
from loss import instance_contrastive_Loss, category_contrastive_loss
from utils import classify
from utils.next_batch import next_batch_gt_5view
from model import Autoencoder, Prediction


class DCPMultiView:
    # Dual contrastive prediction for multi-view
    def __init__(self, config):
        """Constructor.

        Args:
            config: parameters defined in configure.py.
        """
        self._config = config

        self._latent_dim = config['Autoencoder']['arch1'][-1]

        self._dims_view1 = [self._latent_dim] + self._config['Prediction']['arch1']
        self._dims_view2 = [self._latent_dim] + self._config['Prediction']['arch2']
        self._dims_view3 = [self._latent_dim] + self._config['Prediction']['arch3']
        self._dims_view4 = [self._latent_dim] + self._config['Prediction']['arch4']
        self._dims_view5 = [self._latent_dim] + self._config['Prediction']['arch5']
        # 这段代码定义了一个名为DCPMultiView的类，该类具有以下属性和方法：
        # 属性：
        # _config：存储在构造函数中传入的配置参数。
        # _latent_dim：存储在配置参数中自动编码器的最后一层的维度。
        # 方法：
        # __init__：构造函数，接受一个参数config，该参数是在configure.py文件中定义的参数。
        # 参数：
        # config：在configure.py文件中定义的参数。
        # 将传入的配置参数赋值给_config属性。
        # 将自动编码器的最后一层的维度赋值给_latent_dim属性。
        # 根据配置参数和自动编码器的最后一层维度，计算并存储三个视图（view1、view2和view3）的维度列表。
        # 该类的作用是实现多视图的双对比预测，并根据配置参数设置相关属性。
        self.autoencoder1 = Autoencoder(config['Autoencoder']['arch1'], config['Autoencoder']['activations'],
                                        config['Autoencoder']['batchnorm'])
        self.autoencoder2 = Autoencoder(config['Autoencoder']['arch2'], config['Autoencoder']['activations'],
                                        config['Autoencoder']['batchnorm'])
        self.autoencoder3 = Autoencoder(config['Autoencoder']['arch3'], config['Autoencoder']['activations'],
                                        config['Autoencoder']['batchnorm'])
        self.autoencoder4 = Autoencoder(config['Autoencoder']['arch4'], config['Autoencoder']['activations'],
                                        config['Autoencoder']['batchnorm'])
        self.autoencoder5 = Autoencoder(config['Autoencoder']['arch5'], config['Autoencoder']['activations'],
                                        config['Autoencoder']['batchnorm'])

        self.a2b = Prediction(self._dims_view1)
        self.b2a = Prediction(self._dims_view2)
        self.a2c = Prediction(self._dims_view1)
        self.c2a = Prediction(self._dims_view3)
        self.a2d = Prediction(self._dims_view1)
        self.d2a = Prediction(self._dims_view4)
        self.a2e = Prediction(self._dims_view1)
        self.e2a = Prediction(self._dims_view5)

        self.b2c = Prediction(self._dims_view2)
        self.c2b = Prediction(self._dims_view3)
        self.b2d = Prediction(self._dims_view2)
        self.d2b = Prediction(self._dims_view4)
        self.b2e = Prediction(self._dims_view2)
        self.e2b = Prediction(self._dims_view5)

        self.c2d = Prediction(self._dims_view3)
        self.d2c = Prediction(self._dims_view4)
        self.c2e = Prediction(self._dims_view3)
        self.e2c = Prediction(self._dims_view5)

        self.d2e = Prediction(self._dims_view4)
        self.e2d = Prediction(self._dims_view5)

    def train_completegraph_supervised(self, config, logger, accumulated_metrics, x1_train, x2_train, x3_train,
                                       x4_train, x5_train, x1_test, x2_test, x3_test, x4_test, x5_test, labels_train,
                                       labels_test, optimizer,
                                       device):
        """Training the model with complete graph for classification

            Args:
              config: parameters which defined in configure.py.
              logger: print the information.
              accumulated_metrics: list of metrics
              x*_train: training data of view *
              x*_test: test data of view *
              labels_train/test: label of training/test data
              mask *: generate missing data
              optimizer: adam is used in our experiments
              device: to cuda if gpu is used
            Returns:
              classification performance: acc, precision, f-measure

        """
        epochs = config['training']['epoch']
        batch_size = config['training']['batch_size']

        train_view1 = x1_train
        train_view2 = x2_train
        train_view3 = x3_train
        train_view4 = x4_train
        train_view5 = x5_train

        GT = torch.from_numpy(labels_train).long().to(device)
        # classes = np.unique(np.concatenate([labels_train, labels_test])).size  # 查看一共有几个类别
        classes = 3
        flag_gt = True

        for k in range(epochs):
            X1, X2, X3, X4, X5, gt = shuffle(train_view1, train_view2, train_view3, train_view4, train_view5, GT)
            all_ccl, all0, all1, all2, all_icl, map1, map2 = 0, 0, 0, 0, 0, 0, 0

            for batch_x1, batch_x2, batch_x3, batch_x4, batch_x5, gt_batch, batch_No in next_batch_gt_5view(X1, X2, X3,
                                                                                                            X4, X5, gt,
                                                                                                            batch_size):
                z_half1 = self.autoencoder1.encoder(batch_x1)
                z_half2 = self.autoencoder2.encoder(batch_x2)
                z_half3 = self.autoencoder3.encoder(batch_x3)
                z_half4 = self.autoencoder4.encoder(batch_x4)
                z_half5 = self.autoencoder5.encoder(batch_x5)

                # Within-view Reconstruction Loss
                recon1 = F.mse_loss(self.autoencoder1.decoder(z_half1), batch_x1)
                recon2 = F.mse_loss(self.autoencoder2.decoder(z_half2), batch_x2)
                recon3 = F.mse_loss(self.autoencoder3.decoder(z_half3), batch_x3)
                recon4 = F.mse_loss(self.autoencoder4.decoder(z_half4), batch_x4)
                recon5 = F.mse_loss(self.autoencoder5.decoder(z_half5), batch_x5)
                reconstruction_loss = recon1 + recon2 + recon3 + recon4 + recon5

                # Instance-level contrastive loss实例级对比损失
                loss_icl1 = instance_contrastive_Loss(z_half1, z_half2, config['training']['alpha'])
                loss_icl2 = instance_contrastive_Loss(z_half1, z_half3, config['training']['alpha'])
                loss_icl3 = instance_contrastive_Loss(z_half1, z_half4, config['training']['alpha'])
                loss_icl4 = instance_contrastive_Loss(z_half1, z_half5, config['training']['alpha'])
                loss_icl5 = instance_contrastive_Loss(z_half2, z_half3, config['training']['alpha'])
                loss_icl6 = instance_contrastive_Loss(z_half2, z_half4, config['training']['alpha'])
                loss_icl7 = instance_contrastive_Loss(z_half2, z_half5, config['training']['alpha'])
                loss_icl8 = instance_contrastive_Loss(z_half3, z_half4, config['training']['alpha'])
                loss_icl9 = instance_contrastive_Loss(z_half3, z_half5, config['training']['alpha'])
                loss_icl10 = instance_contrastive_Loss(z_half4, z_half5, config['training']['alpha'])

                loss_icl = (
                                       loss_icl1 + 0.1 * loss_icl2 + 0.1 * loss_icl3 + loss_icl4 * 0.1 + loss_icl5 * 0.1 + loss_icl6 * 0.1
                                       + loss_icl7 * 0.1 + loss_icl8 * 0.1 + loss_icl9 * 0.1 + loss_icl10 * 0.1) / 10

                # Cross-view Dual-Prediction Loss跨视图双重预测损失
                a2b, _ = self.a2b(z_half1)
                a2c, _ = self.a2c(z_half1)
                a2d, _ = self.a2b(z_half1)
                a2e, _ = self.a2b(z_half1)
                b2c, _ = self.b2c(z_half2)
                b2d, _ = self.b2c(z_half2)
                b2e, _ = self.b2c(z_half2)
                c2d, _ = self.c2a(z_half3)
                c2e, _ = self.c2a(z_half3)
                d2e, _ = self.c2a(z_half4)

                b2a, _ = self.b2a(z_half2)
                c2a, _ = self.b2a(z_half3)
                d2a, _ = self.b2a(z_half4)
                e2a, _ = self.b2a(z_half5)
                c2b, _ = self.b2a(z_half3)
                d2b, _ = self.b2a(z_half4)
                e2b, _ = self.b2a(z_half5)
                d2c, _ = self.b2a(z_half4)
                e2c, _ = self.b2a(z_half5)
                e2d, _ = self.c2b(z_half5)

                pre1 = F.mse_loss(a2b, z_half2)
                pre2 = F.mse_loss(b2a, z_half1)
                pre3 = F.mse_loss(a2c, z_half3)
                pre4 = F.mse_loss(c2a, z_half1)
                pre5 = F.mse_loss(a2d, z_half4)
                pre6 = F.mse_loss(d2a, z_half1)
                pre7 = F.mse_loss(a2e, z_half5)
                pre8 = F.mse_loss(e2a, z_half1)
                pre9 = F.mse_loss(b2c, z_half3)
                pre10 = F.mse_loss(c2b, z_half2)
                pre11 = F.mse_loss(b2d, z_half4)
                pre12 = F.mse_loss(d2b, z_half2)
                pre13 = F.mse_loss(b2e, z_half5)
                pre14 = F.mse_loss(e2b, z_half2)
                pre15 = F.mse_loss(c2d, z_half4)
                pre16 = F.mse_loss(d2c, z_half3)
                pre17 = F.mse_loss(c2e, z_half5)
                pre18 = F.mse_loss(e2c, z_half3)
                pre19 = F.mse_loss(d2e, z_half5)
                pre20 = F.mse_loss(e2d, z_half4)

                dualprediction_loss = (pre1 + pre2 + pre3 + pre4 + pre5 + pre6 + pre7 + pre8 + pre9 + pre10 + pre11
                                       + pre12 + pre13 + pre14 + pre15 + pre16 + pre17 + pre18 + pre19 + pre20) / 5

                # Category-level contrastive loss类别层次对比损失
                loss_ccl = category_contrastive_loss(torch.cat([z_half1, z_half2, z_half3, z_half4, z_half5], dim=1), gt_batch, classes,
                                                     flag_gt)

                all_loss = loss_icl + reconstruction_loss * config['training']['lambda2'] + loss_ccl
                if k >= config['training']['start_dual_prediction']:
                    all_loss += config['training']['lambda1'] * dualprediction_loss

                optimizer.zero_grad()
                all_loss.backward()
                optimizer.step()

                all_icl += loss_icl.item()
                all_ccl += loss_ccl.item()
                all0 += all_loss.item()
                all1 += recon1.item()
                all2 += recon2.item()
                map1 += pre1.item()
                map2 += pre2.item()
            output = "Epoch : {:.0f}/{:.0f} ===> Reconstruction loss = {:.4f}===> Reconstruction loss = {:.4f} " \
                     "===> Map loss = {:.4f} ===> Map loss = {:.4f} ===> Loss_icl = {:.4e} ===> Los_ccl = {:.4e} ===> All loss = {:.4e}" \
                .format((k + 1), epochs, all1, all2, map1, map2, all_icl, all_ccl, all0)

            if (k + 1) % config['print_num'] == 0:
                logger.info("\033[2;29m" + output + "\033[0m")

            # evalution
            if (k + 1) % config['print_num'] == 0:
                # if True:
                with torch.no_grad():
                    self.autoencoder1.eval(), self.autoencoder2.eval(), self.autoencoder3.eval(), self.autoencoder4.eval(), self.autoencoder5.eval()
                    self.a2b.eval(), self.b2a.eval()
                    self.b2c.eval(), self.c2b.eval()
                    self.a2c.eval(), self.c2a.eval()
                    self.a2d.eval(), self.d2a.eval()
                    self.a2e.eval(), self.e2a.eval()
                    self.b2d.eval(), self.d2b.eval()
                    self.b2e.eval(), self.e2b.eval()
                    self.c2d.eval(), self.d2c.eval()
                    self.c2e.eval(), self.e2c.eval()
                    self.d2e.eval(), self.e2d.eval()


                    # Training data

                    a_latent_eval = self.autoencoder1.encoder(x1_train)
                    b_latent_eval = self.autoencoder2.encoder(x2_train)
                    c_latent_eval = self.autoencoder3.encoder(x3_train)
                    d_latent_eval = self.autoencoder4.encoder(x4_train)
                    e_latent_eval = self.autoencoder5.encoder(x5_train)

                    latent_code_a_eval = torch.zeros(x1_train.shape[0], config['Autoencoder']['arch1'][-1]).to(
                        device)
                    latent_code_b_eval = torch.zeros(x2_train.shape[0], config['Autoencoder']['arch2'][-1]).to(
                        device)
                    latent_code_c_eval = torch.zeros(x3_train.shape[0], config['Autoencoder']['arch3'][-1]).to(
                        device)
                    latent_code_d_eval = torch.zeros(x4_train.shape[0], config['Autoencoder']['arch4'][-1]).to(
                        device)
                    latent_code_e_eval = torch.zeros(x5_train.shape[0], config['Autoencoder']['arch5'][-1]).to(
                        device)


                    latent_fusion_train = torch.cat([latent_code_a_eval, latent_code_b_eval, latent_code_c_eval,latent_code_d_eval,latent_code_e_eval],
                                                    dim=1).cpu().numpy()


                    # Test data
                    a_latent_eval = self.autoencoder1.encoder(x1_test)
                    b_latent_eval = self.autoencoder2.encoder(x2_test)
                    c_latent_eval = self.autoencoder3.encoder(x3_test)
                    d_latent_eval = self.autoencoder4.encoder(x4_test)
                    e_latent_eval = self.autoencoder5.encoder(x5_test)

                    latent_code_a_eval = torch.zeros(x1_test.shape[0], config['Autoencoder']['arch1'][-1]).to(
                        device)
                    latent_code_b_eval = torch.zeros(x2_test.shape[0], config['Autoencoder']['arch2'][-1]).to(
                        device)
                    latent_code_c_eval = torch.zeros(x3_test.shape[0], config['Autoencoder']['arch3'][-1]).to(
                        device)
                    latent_code_d_eval = torch.zeros(x4_test.shape[0], config['Autoencoder']['arch4'][-1]).to(
                        device)
                    latent_code_e_eval = torch.zeros(x5_test.shape[0], config['Autoencoder']['arch5'][-1]).to(
                        device)


                    latent_fusion_test = torch.cat([latent_code_a_eval, latent_code_b_eval, latent_code_c_eval,latent_code_d_eval,latent_code_e_eval],
                                                   dim=1).cpu().numpy()

                    from sklearn.metrics import accuracy_score
                    from sklearn.metrics import precision_score
                    from sklearn.metrics import f1_score

                    label_pre = classify.ave(latent_fusion_train, latent_fusion_test, labels_train)

                    scores = accuracy_score(labels_test, label_pre)

                    precision = precision_score(labels_test, label_pre, average='macro')
                    precision = np.round(precision, 2)

                    f_score = f1_score(labels_test, label_pre, average='macro')
                    f_score = np.round(f_score, 2)

                    accumulated_metrics['acc'].append(scores)
                    accumulated_metrics['precision'].append(precision)
                    accumulated_metrics['f_measure'].append(f_score)
                    logger.info('\033[2;29m Accuracy on the test set is {:.4f}'.format(scores))
                    logger.info('\033[2;29m Precision on the test set is {:.4f}'.format(precision))
                    logger.info('\033[2;29m F_score on the test set is {:.4f}'.format(f_score))

                    self.autoencoder1.train(), self.autoencoder2.train(), self.autoencoder3.train(), self.autoencoder4.train(), self.autoencoder5.train()
                    self.a2b.train(), self.b2a.train()
                    self.b2c.train(), self.c2b.train()
                    self.a2c.train(), self.c2a.train()
                    self.a2d.train(), self.d2a.train()
                    self.a2e.train(), self.e2a.train()
                    self.b2d.train(), self.d2b.train()
                    self.b2e.train(), self.e2b.train()
                    self.c2d.train(), self.d2c.train()
                    self.c2e.train(), self.e2c.train()
                    self.d2e.train(), self.e2d.train()

        return accumulated_metrics['acc'][-1], accumulated_metrics['precision'][-1], accumulated_metrics['f_measure'][
            -1]
