#!/usr/bin/env python
# coding: utf-8

# # Dictionaries, Sets, Tuples
# 
# By [Allison Parrish](http://www.decontextualize.com/)
# 
# ## Dictionaries
# 
# The dictionary is a very useful data structure in Python. The easiest way to conceptualize a dictionary is that it's like a list, except you don't look up values in a dictionary by their index in a sequence---you look them up using a "key," or a unique identifier for that value.
# 
# We're going to focus here just on learning how to get data *out* of dictionaries, not how to build new dictionaries from existing data. We're also going to omit some of the nitty-gritty details about how dictionaries work internally. You'll learn a lot of those details in later courses, but for now it means that some of what I'm going to tell you will seem weird and magical. Be prepared!
# 
# ### Why dictionaries?
# 
# For our purposes, the benefit of having data that can be parsed into dictionaries, as opposed to lists, is that dictionary keys tend to be *mnemonic*. That is, a dictionary key will usually tell you something about what its value is. (This is in opposition to parsing, say, CSV data, where we have to keep counting fields in the header row and translating that to the index that we want.)
# 
# Lists and dictionaries work together and are used extensively to represent all different kinds of data. Often, when we get data from a remote source, or when we choose how to represent data internally, we'll use both in tandem. The most common form this will take is representing a table, or a database, as a *list* of records that are themselves represented as *dictionaries* (mapping the name of the column to the value for that column). We'll see an example of this when we access web APIs in a subsequent tutorial.
# 
# Dictionaries are also good for storing *associations* or *mappings* for quick lookups. For example, if you wanted to write a program that was able to recall the capital city of every US state, you might use a dictionary whose keys are the names of the states, and whose values are the corresponding capitals. Dictionaries are also used for data analysis tasks, like keeping track of how many times a particular token occurs in an incoming data stream.
# 
# ### What dictionaries look like
# 
# Dictionaries are written with curly brackets, surrounding a series of comma-separated pairs of *keys* and *values*. Here's a very simple dictionary, with one key, `Obama`, associated with a value, `Hawaii`:

# In[4]:


{'Obama': 'Hawaii'}


# Here's another dictionary, with more entries:

# In[3]:


{'Obama': 'Hawaii', 'Bush': 'Texas', 'Clinton': 'Arkansas', 'Trump': 'New York'}


# As you can see, we're building a simple dictionary that associates the names of presidents to the home states of those presidents. (This is my version of JOURNALISM.)
# 
# The association of a key with a value is sometimes called a *mapping*. (In fact, in other programming languages like Java, the dictionary data structure is called a "Map.") So, in the above dictionary for example, we might say that the key `Bill Clinton` *maps to* the value `Arkansas`.
# 
# A dictionary is just like any other Python value. You can assign it to a variable:

# In[5]:


president_states = {'Obama': 'Hawaii', 'Bush': 'Texas', 'Clinton': 'Arkansas', 'Trump': 'New York'}


# And that value has a type:

# In[18]:


type(president_states)


# At its most basic level, a dictionary is sort of like a two-column spreadsheet, where the key is one column and the value is another column. If you were to represent the dictionary above as a spreadsheet, it might look like this:
# 
# | key   | value   |
# | ----- | ------- |
# | Obama | Hawaii |
# | Bush | Texas |
# | Clinton | Arkansas |
# | Trump | New York |
# 
# The main difference between a spreadsheet and a dictionary is that dictionaries are *unordered*. For an explanation of this, see below.

# Keys and values in dictionaries can be of any data type, not just strings. Here's a dictionary, for example, that maps integers to lists of floating point numbers:

# In[9]:


{17: [1.6, 2.45], 42: [11.6, 19.4], 101: [0.123, 4.89]}


# > HEAD-SPINNING OPTIONAL ASIDE: Actually, "any type" above is a simplification: *values* can be of any type, but keys must be *hashable*---see [the Python glossary](https://docs.python.org/2/glossary.html#term-hashable) for more information. In practice, this limitation means you can't use lists (or dictionaries themselves) as keys in dictionaries. There are ways of getting around this, though!
# 
# A dictionary can also be empty, containing no key/value pairs at all:

# In[10]:


{}


# ### Getting values out of dictionaries
# 
# The primary operation that we'll perform on dictionaries is writing an expression that evaluates to the value for a particular key. We do that with the same syntax we used to get a value at a particular index from a list. Except with dictionaries, instead of using a number, we use one of the keys that we had specified for the value when making the dictionary. For example, if we wanted to know what Bill Clinton's home state was, or, more precisely, what the value for the key `Clinton` is, we would write this expression:

# In[19]:


president_states["Clinton"]


# Going back to our spreadsheet analogy, this is like looking for the row whose first column is "Clinton" and getting the value from the corresponding second column.

# If we put a key in those brackets that does not exist in the dictionary, we get an error similar to the one we get when trying to access an element of an array beyond the end of a list:

# In[12]:


president_states['Franklin']


# As you might suspect, the thing you put inside the brackets doesn't have to be a string; it can be any Python expression, as long as it evaluates to something that is a key in the dictionary:

# In[17]:


president = 'Obama'
president_states[president]


# You can get a list of all the keys in a dictionary using the dictionary's `.keys()` method:

# In[20]:


president_states.keys()


# That funny-looking `dict_keys(...)` thing isn't *exactly* a list, but it's close enough: you can use it anywhere you would normally use a list, like in a list comprehension:

# In[21]:


[item.upper() for item in president_states.keys()]


# ... or a `for` loop:

# In[22]:


for item in president_states.keys():
    print(item)


# And a list of all the values with the `.values()` method:

# In[15]:


president_states.values()


# If you want a list of all key/value pairs, you can call the `.items()` method:

# In[16]:


president_states.items()


# (The weird list-like things here that use parentheses instead of brackets are called *tuples*---we'll discuss those at a later date.)

# ### Other operations on dictionaries
# 
# [Here's a list of all the methods that dictionaries support](https://docs.python.org/3.6/library/stdtypes.html#mapping-types-dict). I want to talk about a few of these in particular. First, the in operator (which we've used previously to check to see if there's a substring in a string, or a particular item in a list), also works with dictionaries! It checks to see if a particular key exists in the dictionary:

# In[43]:


'Obama' in president_states


# In[44]:


'Franklin' in president_states


# A dictionary can also go in a `for` loop, in the spot between `in` and the colon (where you might normally put a list). If you write a for loop like this, the loop will iterate over each key in the dictionary:

# In[45]:


for item in president_states:
    print(item)


# ### Dictionaries can contain lists and other dictionaries
# 
# Dictionaries are often used to represent *hierarchical* data structures, that is, data structures with a top-down organization. For example, consider a program intended to keep track of a shopping list. In such a program, you might want to categorize grocery items by category, so you might make a dictionary that has a key for each category:

# In[46]:


shopping = {'produce': ['apples', 'oranges', 'spinach', 'carrots'],
            'meat': ['ground beef', 'chicken breast']}


# The `shopping` dictionary above has two keys, whose values are both *lists*. Writing an expression that evaluates to one of these lists is easy, e.g.:

# In[47]:


shopping['meat']


# And you could write a `for` loop to print out the items of one of these lists fairly easily as well, e.g.:

# In[48]:


print("Produce items on your list:")
for item in shopping['produce']:
    print("* " + item)


# Slightly more challenging is this: how do you write an expression that evaluates to (let's say) the *first item* of the list of produce? The trick to this is to remember how indexing syntax works. When you have a pair of square brackets with a single value inside of them, Python looks immediately to the left of those square brackets for an expression that *evaluates to* either a list or a dictionary. For example, in the following expression:

# In[49]:


[5, 10, 15, 20][3]


# ... you can think of Python as looking at this expression from right to left. It sees the `[3]` first and then thinks, "okay, I need to find something that is a list or dictionary directly to the left of this, and grab the third item (index-wise)." In fact, it *does* find a list or a dictionary (i.e., the list `[5, 10, 15, 20]`) and evaluates the entire expression to `20` accordingly.
# 
# With that in mind, let's rephrase the task. I want to get:
# 
# * the first item
# * from the list that is the value for the key `produce`
# * in the dictionary `shopping`
# 
# We can work at this problem by following these instructions and then writing the expression *in reverse*. To get the first item from a list, we write:
# 
#     ????[0] # the first item
#     
# `????` is just a placeholder for the part of the code that we haven't written yet, but we know that it has to be a list. Then, to get the value for the key `produce`:
# 
#     ????["produce"][0] # from the list that is the value for the key `produce`
#     
# Again, `????` is a placeholder, but now we know it has to be a dictionary. The dictionary, of course, is `shopping`, so we can fill that in as the last step:
# 
#     shopping["produce"][0]
#     
# Let's see what that expression evaluates to:

# In[50]:


shopping["produce"][0]


# Exactly right! But let's say we want to take the organization in our dictionary up a notch and create separate categories for fruits and vegetables. One way to do this would be to make the value for the key `produce` be... another dictionary, like so:

# In[51]:


shopping = {'produce': {'fruits': ['apples', 'oranges'], 'vegetables': ['spinach', 'carrots']},
            'meat': ['ground beef', 'chicken breast']}


# This is now a pretty complicated data structure! (Well, not *that* complicated compared to what you'll see, e.g., in responses from web APIs. But it's the most complicated data structure we've made so far.) If we were to draw a schematic of this data structure, it might look something like this:
# 
#     shopping (dictionary)
#         -> produce (dictionary)
#             -> fruits (list)
#             -> vegetables (list)
#         -> meat (list)
#         
# In prose: `shopping` is a variable that contains a dictionary. That dictionary has two keys: `produce`, whose value is itself a dictionary, and `meat`, whose value is a list. (Whew!)
# 
# Given this data structure, let's work through how to do the following tasks:
# 
# * Get a list of all fruits
# * Get a list of all categories of produce
# * Get the first fruit
# * Get the second vegetable
# 
# Getting a list of the fruits requires getting the value for the `fruits` key in the dictionary that is the value for the `produce` key. So we start out with:
# 
#                        ['fruits'] -> Step one
#             ['produce']['fruits'] -> Step two
#     shopping['produce']['fruits'] -> Step three
#     
# The final expression:

# In[52]:


# A list of all fruits
shopping['produce']['fruits']


# Continuing with our tasks:

# In[53]:


# a list of all categories of produce
shopping['produce'].keys()


# In[54]:


# the first fruit
shopping['produce']['fruits'][0]


# In[55]:


# the second vegetable
shopping['produce']['vegetables'][1]


# ### Adding key/value pairs to a dictionary
# 
# Once you've assigned a dictionary to a variable, you can add another key/value pair to the dictionary by assigning a value to a new index, like so:

# In[56]:


president_states['Reagan'] = 'California'


# Take a look at the dictionary to see that there's a new key/value pair in there:

# In[62]:


president_states


# ### On the order of dictionary keys
# 
# So something strange is happening here, and you may have already noticed it. If we write some code that iterates over the keys of a dictionary, the keys show up in one order:

# In[63]:


for item in president_states:
    print(item)


# Whereas if we simply evaluate the dictionary, the keys show up in a different order:

# In[64]:


president_states


# What gives? Here's what's up. Underneath the hood, Python stores the key/value pairs in a dictionary *in the order you added them to the dictionary*. This means that when you add a new item to the dictionary, it will show up *last* when you iterate over the dictionary (or get a list of its keys or values, etc.). However, when you simply evaluate a dictionary, Jupyter Notebook takes it upon itself to display the keys in *alphabetical order* instead. So the order that Jupyter Notebook shows the key/value pairs in is *not* the same as the order for the key/value pairs you would get if you iterated over the list in a for loop.
# 
# To add to the confusion, in previous versions of Python, the order of key/value pairs in a dictionary was *arbitrary* (i.e., deterministic but not repeatable; adding the same items to a dictionary might produce different orderings across Python sessions). And the developers have warned us that this aspect of dictionaries might change in the future ([technical discussion here](https://mail.python.org/pipermail/python-dev/2016-September/146327.html)). So don't rely on the fact that *right now* Python preserves insertion order in dictionaries.

# ### Dictionary keys are unique
# 
# Another important fact about dictionaries is that you can't put the same key into one dictionary twice. If you try to write out a dictionary that has the same key used more than once, Python will silently ignore all but one of the key/value pairs. For example:

# In[65]:


{'a': 1, 'a': 2, 'a': 3}


# Similarly, if we attempt to set the value for a key that already exists in the dictionary (using `=`), we won't add a second key/value pair for that key---we'll just overwrite the existing value:

# In[66]:


test_dict = {'a': 1, 'b': 2}
test_dict['a']


# In[67]:


test_dict['a'] = 100
test_dict['a']


# In the case where a key needs to map to multiple values, we might instead see a data structure in which the key maps to another kind of data structure that itself can contain multiple values, like a list:

# In[82]:


{'a': [1, 2, 3]}


# ## Sets
# 
# The set is our second important data structure. You can think of a set as a kind of list, but with the following caveats:
# 
# 1. Sets don’t maintain the order of objects after you’ve put them in.
# 2. You can’t add an object to a set if it already has an identical object.
# 
# Objects can be added to a set by calling its `.add()` method (as opposed to the `.append()` method used for lists).
# 
# A corollary to item 1 from the list above is that you can’t use the square bracket notation to access a particular element in a set. Once you’ve added an object, the only operations you can do are to check to see if an object is in the set (with the `in` operator), and iterate over all objects in the set (with, for example, `for`). So, for example, to initialize a set:

# In[85]:


emojis = set()


# And then add some items to the set:

# In[88]:


emojis.add("😀")
emojis.add("😜")
emojis.add("😐")


# Evaluating the set shows us its members:

# In[89]:


emojis


# And you can check to see if something is in a set using the `in` operator:

# In[90]:


'😀' in emojis


# In[92]:


'hello' in emojis


# You can write a loop that executes code for every item in a set by putting the set between `in` and the colon in a `for` loop:

# In[94]:


for item in emojis:
    print(item)


# An additional aspect of sets to note from the transcript above: because sets don’t maintain the order of objects, you’ll get the objects back in (seemingly) random order when you iterate over the set. For most applications, this isn’t a problem, but it’s something to keep in mind.
# 
# One thing you'll notice about sets is that you can't add the same item twice. Observe:

# In[95]:


important_numbers = set([5, 10, 15, 20, 25])


# In[96]:


important_numbers


# In[98]:


important_numbers.add(15)


# In[100]:


important_numbers


# For this reason, sets are often used as a way to quickly remove duplicates from a list. All you need to do is pass a list to the `set()` function, and then use the `list()` function to convert the data back into a list. For example:

# In[101]:


letters = ["a", "b", "c", "a", "d", "a", "b", "e"]


# In[104]:


list(set(letters))


# ## Tuples
# 
# Tuples (rhymes with "supple") are data structures very similar to lists. You can create a tuple using parentheses (instead of square brackets, as you would with a list):

# In[105]:


t = ("alpha", "beta", "gamma", "delta")
t


# You can access the values in a tuple in the same way as you access the values in a list: using square bracket indexing 
# syntax. Tuples support slice syntax and negative indexes, just like lists:

# In[106]:


t[-2]


# In[107]:


t[1:3]


# The difference between a list and a tuple is that the values in a tuple can't be changed after the tuple is created. This means, for example, that attempting to .append() a value to a tuple will fail:

# In[108]:


t.append("epsilon")


# Likewise, assigning to an index of a tuple will fail:

# In[ ]:


t[2] = "bravo"


# ### Why tuples? Why now?
# 
# "So," you think to yourself. "Tuples are just like... broken lists. That's strange and a little unreasonable. Why even have them in your programming language?" That's a fair question, and answering it requires a bit of knowledge of how Python works with these two kinds of values (lists and tuples) behind the scenes.
# 
# Essentially, tuples are *faster* and *smaller* than lists. Because lists can be modified, potentially becoming larger after they're initialized, Python has to allocate more memory than is strictly necessary whenever you create a list value. If your list grows beyond what Python has already allocated, Python has to allocate more memory. Allocating memory, copying values into memory, and then freeing memory when it's when no longer needed, are all (perhaps surprisingly) slow processes---slower, at least, than using data already loaded into memory when your program begins.
# 
# Because a tuple can't grow or shrink after it's created, Python knows exactly how much memory to allocate when you create a tuple in your program. That means: less wasted memory, and less wasted time allocating a deallocating memory. The cost of this decreased resource footprint is less versatility.
# 
# Tuples are often called an immutable data type. "Immutable" in this context simply means that it can't be changed after it's created.
# 
# ### Tuples in the standard library
# 
# Because tuples are faster, they're often the data type that gets returned from methods and functions in Python's built-in library. For example, the .items() method of the dictionary object returns a list of tuples (rather than, as you might otherwise expect, a list of lists):

# In[109]:


moon_counts = {'mercury': 0, 'venus': 0, 'earth': 1, 'mars': 2}
moon_counts.items()


# The tuple() function takes a list and returns it as a tuple:

# In[110]:


tuple([1, 2, 3, 4, 5])


# If you want to initialize a new list with with data in a tuple, you can pass the tuple to the list() function:

# In[112]:


list((1, 2, 3, 4, 5))

