from src.graph_utils import analyze_toy_dataset, analyze_karate_club_dataset, analyze_student_dataset, analyze_anybeatAnonymized_dataset
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt
from rich.align import Align


console = Console()
console_width = console.width

def show_menu():
    console.print("\n" * 2)
    
    print("\n" + "=" * console_width)
    console.print(Align.center(Panel(
        "[bold blue]🔍 Social Network Analysis Tool[/bold blue]",
        border_style="blue",
        padding=(1, 3, 1, 3),
        width=40
         
    ))
    )
    console.print("=" * console_width)
    console.print("\n"*2)

    table = Table(show_header=False, box=None)
    table.add_column("Option", style="cyan", no_wrap=True)
    table.add_column("Description", style="white")

    table.add_row("1", "Load and analyze Toy dataset")
    table.add_row("2", "Load and analyze Karate Club dataset")
    table.add_row("3", "Load and analyze Student cooperation dataset")
    table.add_row("4", "Load and analyze Anybeat dataset")
    table.add_row("0", "Quit")
    

    console.print(table)
    console.print("\n")
    return Prompt.ask("[bold cyan]Enter your choice[/bold cyan]", choices=["0", "1", "2", "3", "4"], default="0")



def main():
    while True:
        choice = show_menu()
        if choice == '0':
            console.print("\n[bold blue]Thank you for using Social Network Analysis Tool! See you soon! 👋[/bold blue]\n")
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