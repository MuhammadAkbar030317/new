import sys
sys.path.append(r"C:\Users\User\Desktop\Project_09\Source")


from pipeline import HotelCancellationTrainer


def main():
    trainer = HotelCancellationTrainer()
    trainer.run()


if __name__ == "__main__":
    main()
