from src.graph_utils import analyze_toy_dataset, analyze_karate_club_dataset, analyze_student_dataset, analyze_anybeatAnonymized_dataset

def show_menu():
    print("\n" + "="*50)
    print("Social Network Analysis Tool")
    print("="*50)
    print("1. Load and analyze Toy dataset")
    print("2. Load and analyze Karate Club dataset")
    print("3. Load and analyze Student cooperation dataset")
    print("4. Load and analyze Anybeat dataset")
    print("0. Quit")
    print("="*50)
    return input("Enter your choice (0-4): ")

def main():
    while True:
        choice = show_menu()
        if choice == '0':
            print("\n See you!")
            break
        elif choice == '1':
            analyze_toy_dataset()
        elif choice == '2':
            analyze_karate_club_dataset()
        elif choice == "3":
            analyze_student_dataset()
        elif choice == "4":
           analyze_anybeatAnonymized_dataset()
        else:
            print("\nInvalid choice. Enter a number between 0 and 4.")
        
        input("\nClick enter ton continue...")

if __name__ == "__main__":
    main()