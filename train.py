from model_selection import evaluate_and_register

if __name__ == '__main__':
    best_name, best_score = evaluate_and_register()
    print(f"Best model: {best_name} with validation score {best_score}")