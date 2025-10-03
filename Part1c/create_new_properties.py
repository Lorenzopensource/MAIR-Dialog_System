import csv
import random


# Create new properties for the restaurants
with open('restaurant_info.csv','r') as csvinput:
    with open('restaurant_info_new_properties.csv', 'w') as csvoutput:
        writer = csv.writer(csvoutput, lineterminator='\n')
        reader = csv.reader(csvinput)

        all = []
        row = next(reader)
        row.append("foodquality")
        row.append("crowdness")
        row.append("lengthofstay")
        all.append(row)

        for row in reader:
            row.append(random.choice(["good", "bad"]))
            row.append(random.choice(["busy", "not_busy"]))
            row.append(random.choice(["short", "long"]))
            all.append(row)

        writer.writerows(all)





