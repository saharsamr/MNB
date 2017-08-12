import scipy.io as sio

def convert(data_file_path):
    data = sio.loadmat(data_file_path)
    d1 = data['allSStr']
    d2 = data['labels']

    converted_file = open(data_file_path[:-3]+'dat', 'w')
    count = 0
    for j in range(len(d1[0])):
        # try:
        converted_file.write((str(d2[0][j])+' '+' '.join((d1[0][j][0][i][0] if d1[0][j][0][i] else ' ') for i in range(len(d1[0][j][0]))).encode('utf-8')+'\n'))
        # except UnicodeEncodeError as e:
        #     count += 1
        #     pass
    print count

if __name__ == "__main__":
    convert('unigram_rts.mat')
