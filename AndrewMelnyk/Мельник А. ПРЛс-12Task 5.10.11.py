#Андрій Мельник ПРЛс-12 27.02.12
import nltk
from nltk.corpus import brown
training=brown.tagged_sents(categories='news') #Змінна яка запам'ятовує використання тегів у вже промаркованому тексті
sent=('Stephane Hessel the former French Resistance fighter whose 2010 manifesto Time for Outrage inspired social protesters in Europe dies aged 95').split() #Обрав довільне речення з яким я працюватиму
sent
['Stephane', 'Hessel', 'the', 'former', 'French', 'Resistance', 'fighter', 'whose', '2010', 'manifesto', 'Time', 'for', 'Outrage', 'inspired', 'social', 'protesters', 'in', 'Europe', 'dies', 'aged', '95']
#Використання AffixTagger з різними значеннями affix_length та min_stem_length
affix_tagger=nltk.AffixTagger(training,affix_length=-2,min_stem_length=3)
affix_tagger.tag(sent)
[('Stephane', 'NN'), ('Hessel', 'NN'), ('the', None), ('former', 'NN'), ('French', 'WDT'), ('Resistance', 'NN'), ('fighter', 'NN'), ('whose', 'NN'), ('2010', None), ('manifesto', 'NP'), ('Time', None), ('for', None), ('Outrage', 'NN'), ('inspired', 'VBN'), ('social', 'JJ'), ('protesters', 'NNS'), ('in', None), ('Europe', 'NN'), ('dies', None), ('aged', None), ('95', None)]
affix_tagger=nltk.AffixTagger(training,affix_length=1,min_stem_length=1)
affix_tagger.tag(sent)
[('Stephane', 'NP'), ('Hessel', 'NP'), ('the', 'AT'), ('former', 'IN'), ('French', 'NP'), ('Resistance', 'NP'), ('fighter', 'IN'), ('whose', 'BEDZ'), ('2010', 'CD'), ('manifesto', 'NN'), ('Time', 'AT'), ('for', 'IN'), ('Outrage', 'NP'), ('inspired', 'IN'), ('social', 'NN'), ('protesters', 'NN'), ('in', 'IN'), ('Europe', 'NP'), ('dies', 'NN'), ('aged', 'CC'), ('95', 'CD')]
 
