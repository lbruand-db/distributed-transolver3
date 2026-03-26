"""Deploy a registered Transolver-3 model to a Databricks serving endpoint.

Usage (DAB workflow task):
  python deploy_endpoint.py --catalog ml --schema transolver3 \
                            --model_name transolver3 --endpoint_name transolver3-serving
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from transolver3.serving import deploy_serving_endpoint  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Deploy model to serving endpoint")
    parser.add_argument("--catalog", default="ml")
    parser.add_argument("--schema", default="transolver3")
    parser.add_argument("--model_name", default="transolver3")
    parser.add_argument("--endpoint_name", default="transolver3-serving")
    args = parser.parse_args()

    print(f"Deploying {args.catalog}.{args.schema}.{args.model_name} "
          f"to endpoint '{args.endpoint_name}'...")
    result = deploy_serving_endpoint(
        model_name=args.model_name,
        endpoint_name=args.endpoint_name,
        catalog=args.catalog,
        schema=args.schema,
    )
    print(f"Endpoint ready: {result}")


if __name__ == "__main__":
    main()
