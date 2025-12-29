import csv
import os

PENDING_FILE = "data/pending_flights.csv"
FINAL_FILE = "data/flights.csv"

def main():
    if not os.path.exists(PENDING_FILE):
        print("❌ No pending flights found.")
        return

    os.makedirs("data", exist_ok=True)

    with open(PENDING_FILE, newline="", encoding="utf-8") as f:
        reader = list(csv.reader(f))

    if len(reader) <= 1:
        print("✅ No new records to approve.")
        return

    header = reader[0]
    rows = reader[1:]

    approved = []
    rejected = []

    for i, row in enumerate(rows, 1):
        print("\n--------------------------------")
        print(f"Record #{i}")
        for h, v in zip(header, row):
            print(f"{h:15}: {v}")

        decision = input("Approve this record? (y/n): ").strip().lower()

        if decision == "y":
            approved.append(row)
            print("✅ Approved")
        else:
            rejected.append(row)
            print("❌ Rejected")

    # 写入正式数据
    if approved:
        file_exists = os.path.exists(FINAL_FILE)

        with open(FINAL_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            if not file_exists:
                writer.writerow(header)

            writer.writerows(approved)

    # 重写 pending（只保留没通过的）
    with open(PENDING_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rejected)

    print("\n==============================")
    print(f"Approved: {len(approved)}")
    print(f"Remaining pending: {len(rejected)}")
    print("Done.")

if __name__ == "__main__":
    main()
