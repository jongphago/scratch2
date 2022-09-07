if '__file__' in globals():
    import os
    import sys
    path = os.path.join(os.path.dirname(__file__))
    if path not in sys.path:
        sys.path.append(path)

from ch.ch3 import ex332 as main


if __name__ == '__main__':
    print(f"current file: {main.__name__}")
    print("=" * 19)
    main()
