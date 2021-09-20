from setuptools import setup

setup(
    name='rl_agents',
    version='0.0.1',
    packages=['rl_agents'],
    install_requires=[
        'PyYAML>=5.4.1',
        'scikit-image>=0.18.3',
        'stable-baselines3>=1.2.0',
        'tensorboard>=2.6.0',
    ],
    dev_requires=[
        'pytest >= 6.2.5',
    ]
)