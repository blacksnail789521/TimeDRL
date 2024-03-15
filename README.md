# TimeDRL

Welcome to the official codebase of TimeDRL.  
This project is based on research that has been accepted for publication at the International Conference on Data Engineering (ICDE) 2024.

# Usage

1. Install Python 3.8, and use `requirements.txt` to install the dependencies
   ```
   pip install -r requirements.txt
   ```
2. To execute the script with configuration settings passed via argparse, use:
   ```
   python main.py --...
   ```
   Alternatively, if you prefer to use locally defined parameters to overwrite args for faster experimentation iterations, run:
   ```
   python main.py --overwrite_args
   ```
3. Please refer to `exp_settings_and_results` to see all the experiments' settings and corresponding results.

# Citation

If you find value in this repository, we kindly ask that you cite our paper.

```
@article{chang2023timedrl,
  title={TimeDRL: Disentangled Representation Learning for Multivariate Time-Series},
  author={Chang, Ching and Chan, Chiao-Tung and Wang, Wei-Yao and Peng, Wen-Chih and Chen, Tien-Fu},
  journal={arXiv preprint arXiv:2312.04142},
  year={2023}
}
```

# Contact

If you have any questions or suggestions, please reach out to Ching Chang at [blacksnail789521@gmail.com](mailto:blacksnail789521@gmail.com), or raise them in the 'Issues' section.

# Acknowledgement

This library was built upon the following repositories:

* Time Series Library (TSlib): [https://github.com/thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library)
