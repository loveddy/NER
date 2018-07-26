#coding=utf8
import jieba
# label_list = []
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
# with open('data/train_labeled_data.txt', 'r') as file:
#
#     del_list = []
#     num = -1
#     for line_1 in file:
#         word_list = []
#         word_and_label_list = line_1.strip().split('  ')
#         for word_and_label in word_and_label_list:
#             word = word_and_label.split('/')
#             word_list.append(word[0].decode('gbk'))
#         line = "".join(word_list)
#         if not len([w for w in jieba.cut(line)]) == len(word_list):
#             print word_list
#             print [w.decode('unicode_escape') for w in jieba.cut(line)]



# with open('labeled_data.txt', 'r') as file_1, open('train__labeled_data.txt', 'w') as file_2, open('test_labeled_data.txt', 'w') as file_3:
#     count = 0
#     for line in file_1:
#         count += 1
#         if count <= 17517:
#             file_2.write(line)
#         else:
#             file_3.write(line)
# with open('data/raw/all', 'r') as file, open('data/raw/1temp', 'w') as out:
#     for line in file:
#         if not line == '':
#             del_line = False
#             word_and_label_list = line.strip().split('  ')
#             if len(word_and_label_list) > 1:
#                 for word_and_label in word_and_label_list:
#                     word = word_and_label.split('/')
#                     if len(word[1]) > 2 and ']' not in word[1]:
#                         del_line = True
#                         break
#             else:
#                 del_line = True
#             if not del_line:
#                 temp = ''
#                 for i in range(1, len(word_and_label_list) - 1):
#                     temp += (word_and_label_list[i] + '  ')
#                 temp += word_and_label_list[-1]+'\n'
#                 out.write(temp)




# with open('data/raw/1temp', 'r') as file, open('data/raw/sentence', 'w') as data, open('data/raw/tag', 'w') as tag, open('data/raw/feature', 'w') as feature:
#     for line in file:
#         word_and_label_list = line.strip().split("  ")
#         word_list = []
#         feature_list = []
#         label_list = []
#         start = -1
#         write = True
#         for i, word_and_label in enumerate(word_and_label_list):
#             if '[' in word_and_label and ']' not in word_and_label:
#                 write = False
#                 start = i
#             elif ']' in word_and_label and start > -1 and '[' not in word_and_label:
#                 label = word_and_label.split(']')[1]
#                 if i-start == 1:
#                     word_list.append(word_and_label_list[start].replace('[', '').split('/')[0])
#                     word_list.append(word_and_label_list[i].split('/')[0])
#                     feature_list.append(word_and_label_list[start].replace('[', '').split('/')[1])
#                     feature_list.append(word_and_label_list[i].split(']')[0].split('/')[1])
#                     label_list.append(label+'_b')
#                     label_list.append(label + '_l')
#                 if i-start > 1:
#                     word_list.append(word_and_label_list[start].replace('[', '').split('/')[0])
#                     feature_list.append(word_and_label_list[start].replace('[', '').split('/')[1])
#                     label_list.append(label + '_b')
#                     for j in range(start+1, i):
#                         word_list.append(word_and_label_list[j].split('/')[0])
#                         feature_list.append(word_and_label_list[j].split('/')[1])
#                         label_list.append(label + '_i')
#                     word_list.append(word_and_label_list[i].split('/')[0])
#                     feature_list.append(word_and_label_list[i].split(']')[0].split('/')[1])
#                     label_list.append(label + '_l')
#                 write = True
#                 start = -1
#
#             elif write:
#                 if '[' in word_and_label and ']' in word_and_label:
#                     word_list.append(word_and_label.replace('[', '').split(']')[0].split('/')[0])
#                     label_list.append(word_and_label.split(']', '')[1])
#                     feature_list.append(word_and_label.replace('[', '').split(']')[0].split('/')[1] + '_u')
#                 else:
#                     word_list.append(word_and_label.split('/')[0])
#                     label = word_and_label.split('/')[1]
#                     if label in ['nr', 'nt', 'ns', 'nz', 'i', 'l']:
#                         label += '_u'
#                     else:
#                         label = 'o'
#                     label_list.append(label)
#                     feature_list.append(word_and_label.split('/')[1])
#         if len(word_list) == len(label_list) == len(feature_list):
#             words = ''
#             for i in range(len(word_list)-1):
#                 words += word_list[i] + '\t'
#             words += word_list[-1] + '\n'
#             data.write(words)
#             features = ''
#             for i in range(len(feature_list) - 1):
#                 features += feature_list[i] + '\t'
#             features += feature_list[-1] + '\n'
#             feature.write(features)
#             labels = ''
#             for i in range(len(label_list) - 1):
#                 labels += label_list[i] + '\t'
#             labels += label_list[-1] + '\n'
#             tag.write(labels)

with open('data/raw/tag', 'r') as input, open('data/train/tag', 'w') as train, open('data/test/tag', 'w') as test:
   cut = int(19473 * 0.9)
   now = 0
   for i, line in enumerate(input):
       if now < cut:
           train.write(line)
           now += 1
       else:
           test.write(line)
           now += 1
with open('data/raw/1temp', 'r') as file, open('ckpt/1', 'w') as out:
    list = []
    for line in file:
        temp = line.strip().split('  ')
        for word in temp:
            if ']' in word:
                if word.split(']')[1] not in list:
                    list.append(word.split(']')[1])
    for word in list:
        out.write(word + '\n')