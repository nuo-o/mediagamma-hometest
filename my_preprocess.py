from my_packages import *


def add_date_features(df, date_col):
    hour = []
    #day_of_week = []
    #month = []
    #day = []

    for d in df[date_col]:
        d = pd.to_datetime(d)
        hour.extend([d.hour])
        #day_of_week.extend([d.dayofweek])
        #month.extend([d.month])
        #day.extend([d.day])

    df.loc[:, date_col + '_hour'] = hour
    #df.loc[:, date_col + '_dow'] = day_of_week
    #df.loc[:, date_col + '_day'] = day
    #df.loc[:, date_col + '_month'] = month

    return df


#def add_groupby_features(df):
#    groupby = df.groupby(['ip'])['is_attributed'].sum().reset_index()
#    groupby = groupby.rename(columns = {'is_attributed':'sum_attributed'})
#    df = df.merge(groupby, on='ip', how = 'left')

#    return df


def add_combined_feat(df):
    df['os+device'] = [int(str(os)+str(dev)) for os, dev in zip(df['os'], df['device'])]
    df['os+hour'] = [int(str(os) + str(hour)) for os, hour in zip(df['os'],df['click_time_hour'])]
    return df


def down_sample_train(train, target = 'is_attributed', n=1):
    # down sample click = 0 data so as 'len(click = 0) = n* len(click = 1)'

    click_0 = train[train[target] == 0]
    click_1 = train[train[target] == 1]
    
    sampled_click_index = np.random.choice(click_0.index, int(len(click_1) * n), replace=False)
    sampled_click_df = click_0.loc[sampled_click_index]

    new_train = pd.concat([sampled_click_df, click_1], axis=0)

    return new_train


def create_dummies(df, dummy_variables):
    # one-hot encoding for categorical variables

    for feat in dummy_variables:
        df = pd.concat([df, pd.get_dummies(df[feat], prefix=feat)], axis=1)
        df = df.drop(feat, axis=1)

    return df