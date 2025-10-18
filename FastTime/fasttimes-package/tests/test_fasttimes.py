from setuptools import setup, find_packages

setup(
    name='fasttimes',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        # 在这里列出你的依赖包，例如：
        # 'numpy',
    ],
    entry_points={
        'console_scripts': [
            # 如果你有命令行工具，可以在这里定义
            # 'your_command=your_module:main_function',
        ],
    },
)