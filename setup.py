from setuptools import setup

setup(
    name='rl_agents',
    version='0.0.1',
    packages=['rl_agents'],
    install_requires=[
        'gym >= 0.19.0',
        'matplotlib >= 3.4.3',
        'numpy >= 1.21.2',
        'scikit-image >= 0.18.3',
        'torch >= 1.9.0',
        'torchvision >= 0.10.0',
        'uuid >= 1.30'
    ],
    dev_requires=[
        'pytest >= 6.2.5',
        'stable-baselines >= 2.10.2',
    ]
)