from setuptools import setup

setup(
    name='rl_agents',
    version='0.0.1',
    packages=['rl_agents'],
    install_requires=[
        'pyvirtualdisplay>=2.2',
        'tf_agents >= 0.9.0',
        'tensorflow >= 2.6.0',
        'pillow >= 8.3.2',
        'matplotlib >= 3.4.2',
        'scikit-image >= 0.18.3',
    ],
    dev_requires=[
        'pytest >= 6.2.5',
        'stable-baselines >= 2.10.2',
    ]
)