# mask_test = get_mask(3, 20, 0)
# print(mask_test)

#
# parser = argparse.ArgumentParser()
# parser.add_argument('--missing_rate', type=float, default='0', help='missing rate')
# args = parser.parse_args()
# x = args.missing_rate

# b_train = np.ones(4)

# mask_test = get_mask(3, 4, x)
# x1_train = b_train * mask_test[:, 0][:, np.newaxis]
# x2_train = b_train * mask_test[:, 1][:, np.newaxis]
# x3_train = b_train * mask_test[:, 2][:, np.newaxis]
# print(b_train)
# print(x1_train)
# print(x3_train)
# print(type(x3_train))
# print(type(b_train))
# print(mask_test)

# print(y)
# configs = dict(
#             missing_rate=0,
#             seed=4,
#             view=2,
#             training=dict(
#                 lr=1.0e-4,
#                 start_dual_prediction=500,
#                 batch_size=256,
#                 epoch=1000,
#                 alpha=10,
#                 lambda2=0.1,
#                 lambda1=0.1,
#             ),
#             Autoencoder=dict(
#                 view_size=2,
#                 arch1=[1984, 1024, 1024, 1024, 128],
#                 arch2=[512, 1024, 1024, 1024, 128],
#                 activations1='relu',
#                 activations2='relu',
#                 batchnorm=True,
#             ),
#             Prediction=dict(
#                 arch1=[128, 256, 128],
#                 arch2=[128, 256, 128],
#             ))
# y = configs['Autoencoder']['arch1'][-1]
# print(y)



















