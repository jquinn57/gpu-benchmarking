from memryx import Benchmark
filename = './dfp_models/yolov5n-SiLU-416.dfp'

# when doing the test with two calls to bm.run inside the same context
# sometimes (typcially 3 out of 16) times the FPS drops by a factor of 2.7
def test_bad():
    with Benchmark(dfp=filename) as bm:
        _, latency_ms, _ = bm.run(threading=False)
        _, _, fps = bm.run(frames=1000, threading=True)

    print(f'FPS: {fps}, Latency: {latency_ms}')

# when doing the test this way the resutls seem to be consistent
def test_good():
    with Benchmark(dfp=filename) as bm:
        _, latency_ms, _ = bm.run(threading=False)

    with Benchmark(dfp=filename) as bm:
        _, _, fps = bm.run(frames=1000, threading=True)

    print(f'FPS: {fps}, Latency: {latency_ms}')


if __name__ == '__main__':

    print('\nTest Bad:')
    for n in range(16):
        test_bad()

    print('Test Good:')
    for n in range(16):
        test_good()