import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# 设置全局字体为Times New Roman
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.size'] = 12  # 或者您想要的任何大小
# plt.rcParams.update({'font.size': 12})

with open('/workspaces/tssl/uncertainty_test/softplus.txt') as f:
    data_origin = f.readlines()
    for i in range(len(data_origin)):
        data_origin[i] = data_origin[i].strip('\n')
        data_origin[i] = float(data_origin[i])
with open('/workspaces/tssl/tlstm_uncer_data/locata/snr_-5.txt') as f:
    data_snr_2 = f.readlines()
    for i in range(len(data_snr_2)):
        data_snr_2[i] = data_snr_2[i].strip('\n')
        data_snr_2[i] = float(data_snr_2[i])
with open('/workspaces/tssl/tlstm_uncer_data/locata/snr_-10.txt') as f:
    data_snr_3 = f.readlines()
    for i in range(len(data_snr_3)):
        data_snr_3[i] = data_snr_3[i].strip('\n')
        data_snr_3[i] = float(data_snr_3[i])
with open('/workspaces/tssl/tlstm_uncer_data/locata/snr_-15.txt') as f:
    data_snr_4 = f.readlines()
    for i in range(len(data_snr_4)):
        data_snr_4[i] = data_snr_4[i].strip('\n')
        data_snr_4[i] = float(data_snr_4[i])
with open('/workspaces/tssl/4mic_test_uncer/snr_5.txt') as f:
    data_snr_5 = f.readlines()
    for i in range(len(data_snr_5)):
        data_snr_5[i] = data_snr_5[i].strip('\n')
        data_snr_5[i] = float(data_snr_5[i])

sns.kdeplot(
    data_origin,
    bw_adjust=0.5,
    fill=True,
    color='blue',
    # color='gray',
    label='in-distribution',
    alpha=0.1
)
# sns.kdeplot(
#     data_snr_2,
#     bw_adjust=0.5,
#     fill=True,
#     color='red',
#     label='OOD (SNR=-5 dB)',
#     alpha=0.1
# )
# sns.kdeplot(
#     data_snr_3,
#     bw_adjust=0.5,
#     fill=True,
#     color='yellow',
#     label='OOD (SNR=-10 dB)',
#     alpha=0.1
# )
# sns.kdeplot(
#     data_snr_4,
#     bw_adjust=0.5,
#     fill=True,
#     color='green',
#     label='OOD (SNR=-15 dB)',
#     alpha=0.1
# )
# sns.kdeplot(
#     data_snr_5,
#     bw_adjust=0.5,
#     fill=True,
#     color='purple',
#     label='OOD (SNR=-5 dB)',
#     alpha=0.1
# )

# plt.gca().get_yaxis().set_visible(False)
# NOTE: For the non-4mic or 8mic dataset.
plt.xlabel('Uncertainty (a.u.)', fontsize=16)
plt.ylabel('Density (a.u.)', fontsize=16)
# plt.tick_params(axis='both',which='major',labelsize=14)
plt.legend(fontsize=14)  # 显示图例
# plt.xlim(left=0.3, right=1)
plt.xticks(fontsize=12)
# plt.xticks([])
plt.yticks([])
# 显示图表
plt.savefig('/workspaces/tssl/uncertainty_test/tcrnn_simulate_uncer_dis.pdf', dpi=300, bbox_inches='tight')
# plt.show()
# NOTE: For the 4mic or 8mic dataset.
# plt.xlabel('Uncertainty (a.u.)')
# plt.ylabel('Density (a.u.)')
# # plt.tick_params(axis='both',which='major',labelsize=14)
# plt.legend()  # 显示图例
# plt.xlim(left=0.3, right=1)
# plt.xticks()
# # plt.xticks([])
# plt.yticks([])
# # 显示图表
# plt.savefig('/workspaces/tssl/3rdRevision/tlstm_locata_uncer_dis.pdf', dpi=300, bbox_inches='tight')
# plt.show()
