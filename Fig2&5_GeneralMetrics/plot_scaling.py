import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
def plot_scale(input_type, input_color, input_label_list, file_name):

    def linear_regre(x, y):
        x =  np.array(x)
        x_reshaped = x.reshape(-1, 1)
        model = LinearRegression()
        model.fit(x_reshaped, y)
        y_pred = model.predict(x_reshaped)
        slope = model.coef_[0]
        r_squared = r2_score(y, y_pred)
        return slope, r_squared

    amp_chain = np.load('scaling/'+input_type+'_amplification.npy')
    amp_data = np.load(file_name +input_type+'_amplification_ER.npy')
    paths = amp_data[0]
    means = amp_data[1]
    stds = amp_data[2]

    if len(paths) > 8:
        slope, r_square = linear_regre(paths[1:10], np.log(amp_chain[1:10]))

        plt.figure(figsize=(4, 3))
        ax = plt.axes()
        
        ax.plot(paths[1:10], means[1:10], linestyle='--', linewidth = 3,
                label=fr'${input_label_list[0]} \ for \ ER$', color=input_color)
        ax.fill_between(paths[1:10], np.array(means[1:10]) - np.array(stds[1:10]), 
                        np.array(means[1:10]) + np.array(stds[1:10]), color=input_color, alpha=0.3)
        ax.plot(paths[1:10], amp_chain[1:10], marker = 'o', markersize = 10,
                label=fr'${input_label_list[0]} \ for \ chain$', 
                color=input_color, linewidth=3,  alpha=0.8)
        ax.plot(paths[1:10], np.power((1 + 1/1), -paths[1:10]), 
                label=r'$Leading \ term$', color='black', alpha=1.0, 
                linestyle='-', linewidth=3) # , marker='^', markersize=8
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_yscale('log')
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10, subs=[10], numticks=10))
        ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())
        ax.tick_params(axis='both', which='major', labelsize=20)  # Major ticks


        ax.set_xticks(np.arange(1, 10), np.arange(1, 10))

        plt.tight_layout()
        
        plt.savefig('figures/scaling/amp_' + input_type + '.pdf', dpi=300)
        plt.show()

    else:
        plt.figure(figsize=(4, 3))
        ax = plt.axes()
        ax.plot(paths[1:], means[1:], '-o', label=fr'${input_label_list[0]} \ for \ ER$', color=input_color)
        ax.fill_between(paths[1:], np.array(means[1:]) - np.array(stds[1:]), np.array(means[1:]) + np.array(stds[1:]), color=input_color, alpha=0.3)
        ax.plot(paths[1:], amp_chain[1: len(paths[1:])+1], label = fr'${input_label_list[0]} \ for \ chain$', color = input_color, linewidth = 3)

        ax.plot(paths[1:], np.power((1 + 1/1), -paths[1:]) , label=r'$Leading \ term$', color='lightyellow', alpha = 1.0, linestyle='-', linewidth=3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel(r'$distance \ d$', fontsize = 15)
        ax.set_ylabel(r'$amplification$', fontsize = 15)
        ax.set_yscale('log', base=np.e)
        def format_func(value, tick_number):
            return f'{value:.2e}'  
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_func))
        ax.set_xticks(np.arange(1,len(paths[1:])+1), np.arange(1,len(paths[1:])+1))

        plt.tight_layout()

        plt.savefig('figures/scaling/amp_' + input_type + '.pdf', dpi=300)
        plt.show()

    peak_chain = np.load('scaling/'+input_type+'_peak_response.npy')
    peak_data = np.load(file_name+input_type+'_peak_response_ER.npy')
    paths = peak_data[0]
    means = peak_data[1]
    stds = peak_data[2]

    if len(paths) > 8:
        slope, r_square = linear_regre(paths[1:10], np.log(peak_chain[1:10]))

        plt.figure(figsize=(4, 3))
        ax = plt.axes()
        
        ax.plot(paths[1:10], means[1:10], label=fr'${input_label_list[1]} \ for \ ER$', color=input_color, linestyle = '--', linewidth = 3)
        ax.fill_between(paths[1:10], np.array(means[1:10]) - np.array(stds[1:10]), np.array(means[1:10]) + np.array(stds[1:10]), color=input_color, alpha=0.3)
        ax.plot(paths[1:10], peak_chain[1:10], label = fr'${input_label_list[1]} \ for \ chain$', color = input_color, linewidth = 3, alpha = 0.8,marker='o', markersize=10)
        ax.plot(paths[1:10], np.power((1 + 1/1), -paths[1:10]) , label=r'$Leading \ term$', color='black', alpha = 1.0, linestyle='-', linewidth=3)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)


        ax.set_yscale('log', base=np.e)


        print('slope:', fr'$Slope= {slope:.2f}$')
        ax.set_yscale('log')
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10, subs=[10], numticks=10))
        ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())
        ax.tick_params(axis='both', which='major', labelsize=20)  # Major ticks


        ax.set_xticks(np.arange(1,10), np.arange(1,10))

        plt.tight_layout()
        
        plt.savefig('figures/scaling/peak_' + input_type + '.pdf', dpi=300)
        plt.show()
    
    else:
        plt.figure(figsize=(4, 3))
        ax = plt.axes()
        ax.plot(paths[1:], means[1:], '-o', label=fr'${input_label_list[1]} \ for \ ER$', color=input_color)
        ax.fill_between(paths[1:], np.array(means[1:]) - np.array(stds[1:]), np.array(means[1:]) + np.array(stds[1:]), color=input_color, alpha=0.3)
        ax.plot(paths[1:], peak_chain[1:len(paths[1:]) +1], label = fr'${input_label_list[1]} \ for \ chain$', color = input_color, linewidth = 3)

        ax.plot(paths[1:], np.power((1 + 1/1), -paths[1:]) , label=r'$Leading \ term$', color='gray', alpha = 0.5, linestyle='-', linewidth=3)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel(r'$distance \ d$', fontsize = 15)
        ax.set_ylabel(r'$peak \ response$', fontsize = 15)
        ax.set_yscale('log', base=np.e)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_func))
        ax.set_xticks(np.arange(1,len(paths[1:])+1), np.arange(1,len(paths[1:])+1))

        plt.tight_layout()


        plt.savefig('figures/scaling/peak_' + input_type + '.pdf', dpi=300)
        plt.show()

    if input_type != "square":

        time_chain = np.load('scaling/'+input_type+'_time_constant.npy')
        time_data = np.load(file_name+input_type+'_time_constant_ER.npy')
        paths = time_data[0]
        means = time_data[1] 
        stds = time_data[2]

        if len(paths) > 8:
            slope, r_square = linear_regre(paths[1:10], (time_chain[1:10]))

            plt.figure(figsize=(4, 3))
            ax = plt.axes()
            
            ax.plot(paths[1:10], means[1:10], label=fr'{input_label_list[2]}' +'$ \ for \ ER$', color=input_color, linestyle = '--', linewidth = 3)
            ax.fill_between(paths[1:10], np.array(means[1:10]) - np.array(stds[1:10]), np.array(means[1:10]) + np.array(stds[1:10]), color=input_color, alpha=0.3)
            ax.plot(paths[1:10], time_chain[1:10], label = fr'{input_label_list[2]}'+'$ \ for \ chain$', color = input_color, linewidth = 3,  alpha = 0.8, marker = 'o', markersize  = 10)

            ax.plot(paths[1:10], (1 / (1+1))* paths[1:10], label=r'$Leading \ term$', color='black', alpha = 1.0, linewidth=3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)


            ax.set_xticks(np.arange(1,10), np.arange(1,10))
            ax.tick_params(axis='both', which='major', labelsize=20)  # Major ticks


            print('slope:', fr'$Slope= {slope:.2f}$')
            ax.legend(frameon=False, prop={'size': 12}, loc = 'upper left')
            plt.tight_layout()
            
            plt.savefig('figures/scaling/time_' + input_type + '.pdf', dpi=300)
            plt.show()
        else:
            plt.figure(figsize=(4, 3))
            ax = plt.axes()
            ax.plot(paths[1:], means[1:], '-o', label=fr'{input_label_list[2]}' +'$ \ for \ ER$', color=input_color)
            ax.fill_between(paths[1:], np.array(means[1:]) - np.array(stds[1:]), np.array(means[1:]) + np.array(stds[1:]), color=input_color, alpha=0.3)
            ax.plot(paths[1:], time_chain[1:len(paths[1:])+1], label = fr'{input_label_list[2]}'+'$ \ for \ chain$', color = input_color, linewidth = 3)

            ax.plot(paths[1:], (1 / (1+1))* paths[1:], label=r'$Leading \ term$', color='gray', alpha = 0.5, linestyle='-', linewidth=3)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_xlabel(r'$distance \ d$', fontsize = 15)
            ax.set_ylabel(r'$time$', fontsize = 15)
            ax.set_xticks(np.arange(1,len(paths[1:])+1), np.arange(1,len(paths[1:])+1))

            ax.legend(frameon=False, prop={'size': 10}, loc = 'upper left')
            plt.tight_layout()


            plt.savefig('figures/scaling/time_' + input_type + '.pdf', dpi=300)
            plt.show()

