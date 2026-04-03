import time
from datetime import datetime


number_of_steps = 8
for i in range(number_of_steps):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] Step {i + 1}/{number_of_steps}: job is still running...")
    time.sleep(60)

print("Done. This demo job finished after about 3 minutes.")
