from setuptools import setup, find_packages

setup(
    name='DeepSurrogate_model',  
    version='0.1.0',   # 시작 버전
    packages=find_packages(),  

    install_requires=[
        'tensorflow>=2.0.0',
        'numpy',
        'scipy',
        'pandas',
        'scikit-learn',
        'statsmodels',
        'tqdm'
    ],

    author='Yeseul Jeon',
    author_email='jeons9677@gmail.com',

    description='DeepSurrogate: An interpretable XAI system for functional surrogate modeling.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',

    url='https://github.com/jeon9677/DeepSurrogate_model',  

    license='MIT',

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics'
    ],

    python_requires='>=3.7',
)
