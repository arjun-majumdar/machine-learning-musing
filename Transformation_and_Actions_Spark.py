'''
Transformations and Actions in Apache Spark using Python:

Spark Transformations-

1.) map()- This transformation applies changes to each line of the RDD and
returns the transformed RDD as iterable of iterables i.e., each line is
equivalent to an iterable and the entire RDD is itself a list.

2.) flatMap()- This transformation applies changes to each line in the same way
as map() but the returnable is NOT an iterable of iterables but, it is only an
iterable holding the entire RDD contents. 

3.) filter()
4.) sample()
5.) union()
6.) intersection()
7.) distinct()
8.) join()


Spark Actions-

1.) reduce()             
2.) collect()               
3.) count()
4.) first()    
5.) takeSample(withReplacement, num, [seed])


Refer-
https://www.dezyre.com/apache-spark-tutorial/pyspark-tutorial
'''

# Read a text file-
confused_RDD = sc.textFile("/home/arjun/Documents/Programs/Apache_Spark_Resources/confused.txt")

# Check it's contents by displaying the first 5 lines-
>>> confused_RDD.take(5)
# ['Confusion is the inability to think as clearly or quickly as you normally do.', '', 'You may  have difficulty paying attention to anything , remembering anyone, and making decisions.', '', 'Confusion may come to anyone early or late phase of the life, depending on the reason behind it .']


# Split each string on space and analyse the structure of the text file-
confused_map = confused_RDD.map(lambda line: line.split(" "))

# Print first 5 lines of each line of string being split on space-
confused_map.take(5)
[['Confusion', 'is', 'the', 'inability', 'to', 'think', 'as', 'clearly', 'or', 'quickly', 'as', 'you', 'normally', 'do.'], [''], ['You', 'may', '', 'have', 'difficulty', 'paying', 'attention', 'to', 'anything', ',', 'remembering', 'anyone,', 'and', 'making', 'decisions.'], [''], ['Confusion', 'may', 'come', 'to', 'anyone', 'early', 'or', 'late', 'phase', 'of', 'the', 'life,', 'depending', 'on', 'the', 'reason', 'behind', 'it', '.']]

# From the output, it is evident that each line is a separate iterable of words
# which is contained in another iterable (list) i.e. iterable of iterables.

'''
# OR, you can also do-
for split_line in confused_map.take(5):
    print(split_line)
'''




# Comparing 'flatMap' transformation as compared to 'Map' transformation-
confused_flatMap = confused_RDD.flatMap(lambda line: line.split(" "))

# Print the first 5 iterables- 
confused_flatMap.take(5)
# ['Confusion', 'is', 'the', 'inability', 'to']

# The output shows that each word is now acting as a single  line, i.e., it is
# now an iterable of strings.




# 'filter()' Transformation in Spark:
# This transformation is used to reduce the old RDD based on some condition.
# Let’s try to find out the lines having the term 'confusion' in it in
# 'confused_RDD'-
filter_confusion_word = confused_RDD.filter(lambda line: ("confus" in line.lower()))

filter_confusion_word
# PythonRDD[6] at RDD at PythonRDD.scala:53

filter_confusion_word.count()
# 7

# In this output, we have found that there 7 lines having the word 'confusion'
# in them but to find out those lines, we can use the 'collect()' action in
# Spark as shown below-
filter_confusion_word.collect()
# ['Confusion is the inability to think as clearly or quickly as you normally do.', 'Confusion may come to anyone early or late phase of the life, depending on the reason behind it .', 'Many times, confusion lasts for a very short span and goes away.', 'Confusion is more common in people who are in late stages of the life and often occurs when you have stayed in hospital.', 'Some confused people may have strange or unusual behavior or may act aggressively.', 'A good way to find out if anyone is confused is to question the person their identity i.e. name, age, and the date.', 'If they are little not sure or unable to answer correctly, they are confused']


# Another example using the CHANGES.txt file from the sparke.
# The task is to include only those commits that are done by “Tathagata Das” in
# spark module.

# Read text file-
changes_RDD = sc.textFile("/home/arjun/Documents/Programs/Apache_Spark_Resources/CHANGES.txt")

# Filter commits done by "Das"-
>>> Das_changes = changes_RDD.filter(lambda line: "tathagata.das1565@gmail.com" in line.lower())

# Count 
Das_changes.count()
# 61

# You can print first three lines-
# Das_changes.take(3)


# Similarly, we can see the number of changes made by another developer named
# “Ankur Dave”-
ankur_changes = changes_RDD.filter(lambda line: "ankurdave@gmail.com" in line.lower())

ankur_changes.count()
# 28




'''
sample (withReplacement, fraction, seed)-

This transformation is used to sample RDD from a larger RDD.
It is frequently used in Machine learning operations where a sample of the
dataset needs to be taken.

The fraction means the percentage of the total data, you want to take the
sample from.
'''
# Let’s sample the confused_RDD with 50% of it allowing for replacement-
sampled_confusion = confused_RDD.sample(True, 0.5, 10) # True means sample WITH REPLACEMENT

# Count the number of samples-
sampled_confusion.count()
# 8

# To print all of the 8 samples, use-
sampled_confusion.collect()
# ['', '', 'Confusion may come to anyone early or late phase of the life, depending on the reason behind it .', '', 'Other times, it may be permanent and has no cure. It may have association with delirium or dementia. ', '', '', 'If they are little not sure or unable to answer correctly, they are confused']




'''
union() Transformation in Spark-
'union()' is used to merge two RDDs together if they have the same structure.

Example- A class has two students named, Abhay and Ankur whose marks have to be
combined to get the marks of the entire class. So, here’s how you can do it-
'''
marks_abhay = [("physics", 85), ("maths", 89), ("chemistry", 83)]
marks_ankur = [("physics", 55), ("maths", 91), ("chemistry", 73)]

abhay = sc.parallelize(marks_abhay)
ankur = sc.parallelize(marks_ankur)

abhay.union(ankur).collect()
# [('physics', 85), ('maths', 89), ('chemistry', 83), ('physics', 55), ('maths', 91), ('chemistry', 73)]




'''
join() Transformation in Spark-
This transformation joins two RDDs based on a common key.

Example- In continuation to the above example of union, you can combine the
marks of Abhay and Ankur based on each subject as follows-
'''
subject_wise_marks = abhay.join(ankur)

subject_wise_marks.collect()
# [('physics', (85, 55)), ('maths', (89, 91)), ('chemistry', (83, 73))]




"""
intersection() Transformation-
'intersection()' gives you the common terms or objects from the two RDDS.

Example: Let’s find out the players who are both good cricketers as well as
toppers of the class.
"""
Cricket_team = ["sachin","abhay","michael","rahane","david","ross","raj","rahul","hussy","steven","sourav"]

Toppers = ["rahul","abhay","laxman","bill","steve"]

cricket_RDD = sc.parallelize(Cricket_team)
toppers_RDD = sc.parallelize(Toppers)

toppers_and_crickets = cricket_RDD.intersection(toppers_RDD)

toppers_and_crickets.collect()
# ['rahul', 'abhay']




'''
distinct()  Transformation in Spark-
This transformation is used to get rid of any ambiguities.
As the name suggests it picks out the lines from the RDD that are unique.

Example- Suppose that there are various movie nominations in different
categories. We want to find out, how many movies are nominated overall-
'''
best_story = ["movie1","movie3","movie7","movie5","movie8"]
best_direction = ["movie11","movie1","movie5","movie10","movie7"]
best_screenplay = ["movie10","movie4","movie6","movie7","movie3"]

story_rdd = sc.parallelize(best_story)
direction_rdd = sc.parallelize(best_direction)
screen_rdd = sc.parallelize(best_screenplay)

total_nominations_rdd = story_rdd.union(direction_rdd).union(screen_rdd)

total_nominations_rdd.collect()
# ['movie1', 'movie3', 'movie7', 'movie5', 'movie8', 'movie11', 'movie1', 'movie5', 'movie10', 'movie7', 'movie10', 'movie4', 'movie6', 'movie7', 'movie3']

unique_movies_rdd = total_nominations_rdd.distinct()

unique_movies_rdd.collect()
# ['movie10', 'movie8', 'movie5', 'movie1', 'movie4', 'movie7', 'movie3', 'movie11', 'movie6']


