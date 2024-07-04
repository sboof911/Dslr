from describe import describe

class SortingHat:
    def __init__(self):
        pass
    
    def LogisticRegression(self, Arithmancy, Astronomy, Herbology, DefenseAgainsttheDarkArts, Divination, MuggleStudies, AncientRunes, HistoryofMagic, Transfiguration,
                           Potions, CareofMagicalCreatures, Charms, Flying):
        return 

if __name__ == "__main__":
    import sys, os

    try:
        if len(sys.argv) != 2:
            raise Exception("Enter just the filepath as an arguments!")
        if not sys.argv[1].endswith(".csv"):
            raise Exception(f"The file '{sys.argv[1]}' must be a csv file.")
        if not os.path.exists(sys.argv[1]):
            raise Exception(f"The file '{sys.argv[1]}' does not exist.")

    except Exception as e:
        print(e)