def get_default_config(data_name):
    if data_name in ['MRI']:
        """The default configs."""
        return dict(
            type='CG',  # other: CV
            view=5,
            Autoencoder=dict(
                arch1=[5, 1024, 1024, 1024, 128],
                arch2=[16, 1024, 1024, 1024, 128],
                arch3=[7, 1024, 1024, 1024, 128],
                arch4=[3, 1024, 1024, 1024, 128],
                arch5=[90, 1024, 1024, 1024, 128],
                activations='relu',
                batchnorm=True,
            ),
            Prediction=dict(
                arch1=[128, 256, 128],
                arch2=[128, 256, 128],
                arch3=[128, 256, 128],
                arch4=[128, 256, 128],
                arch5=[128, 256, 128],
            ),
            training=dict(
                lr=1.0e-4,
                start_dual_prediction=0,
                batch_size=256,
                epoch=150,
                alpha=10,
                lambda2=0.1,
                lambda1=0.1
            ),
            seed=8,
        )
