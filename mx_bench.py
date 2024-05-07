from memryx import Benchmark
import argparse
import yaml
import pprint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to config yaml", default="config_mx3.yaml")
    args = parser.parse_args()

    with open(args.config) as fp:
        config = yaml.safe_load(fp)
    pprint.pprint(config)

    fps_all = {}
    for n in range(10):
        print()
        for model in config["models"]:
            model_config = config["models"][model]

            # Do a seperate latency measurement
            with Benchmark(dfp=model_config["filename"], verbose=2, chip_gen=3.1) as bm:
                #_, data["latency_ms"], _ = bm.run(threading=False)
                _, _, fps_bm = bm.run(frames=1000)
                fps_bm = int(round(fps_bm))

            if model in fps_all:
                fps_all[model].append(fps_bm)
            else:
                fps_all[model] = [fps_bm]
            print(f'{model}, FPS: {fps_all[model]}')


if __name__ == "__main__":
    main()
