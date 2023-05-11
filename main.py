import dtlpy as dl
import json
import os
import random
import shutil


def split_training_dataset(train_dataset_dir='trainingSample'):
    # Path to the new /training and /validation directories
    training_dir = f'prepared_{train_dataset_dir}/training'
    validation_dir = f'prepared_{train_dataset_dir}/validation'

    # Create the /training and /validation directories if they don't exist
    os.makedirs(training_dir, exist_ok=True)
    os.makedirs(validation_dir, exist_ok=True)

    # Get a list of subdirectories (0-9) in the /trainSet directory
    subdirectories = [f for f in os.listdir(train_dataset_dir) if os.path.isdir(os.path.join(train_dataset_dir, f))]

    # Iterate through each subdirectory
    for subdir in subdirectories:
        subdir_path = os.path.join(train_dataset_dir, subdir)

        # Get a list of image files in the subdirectory
        image_files = [f for f in os.listdir(subdir_path) if os.path.isfile(os.path.join(subdir_path, f))]

        # Calculate the number of images for training and validation
        num_images = len(image_files)
        ratio = 0.8

        num_training = int(ratio * num_images)
        num_validation = num_images - num_training

        # Shuffle the list of image files
        random.shuffle(image_files)

        # Move the images to the appropriate directories
        for i, image_file in enumerate(image_files):
            src = os.path.join(subdir_path, image_file)
            if i < num_training:
                dst = os.path.join(training_dir, subdir, image_file)
            else:
                dst = os.path.join(validation_dir, subdir, image_file)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)


def upload_train_dataset_without_json(project: dl.Project, prepared_train_dataset_dir='prepared_trainingSample'):
    # Get MNIST dataset
    try:
        dataset = project.datasets.create(dataset_name='MNIST_Dataset')
    except dl.exceptions.BadRequest:
        dataset = project.datasets.get(dataset_name='MNIST_Dataset')

    # Upload items to the dataset
    dataset.to_df()
    _ = dataset.items.upload(local_path=f'{prepared_train_dataset_dir}/*', overwrite=False)
    dataset.add_labels(label_list=[str(i) for i in range(10)])

    # Update dataset metadata
    # dataset.metadata['system']['subsets'] = {
    #     'train': json.dumps(dl.Filters(field='dir', values='/train/*').prepare()),
    #     'validation': json.dumps(dl.Filters(field='dir', values='/validation/*').prepare()),
    # }
    # dataset.update(system_metadata=True)

    # Upload labels
    pages = dataset.items.list()
    item_builders = list()

    for page in pages:
        for item in page:
            builder = item.annotations.builder()
            item_label = item.filename.split('/')[2]
            builder.add(annotation_definition=dl.Classification(label=item_label))
            item_builders.append(builder)

    for builder in item_builders:
        builder.item.annotations.upload(annotations=builder)

    dataset.open_in_web()


# TODO: implement
def upload_train_dataset_with_json(project: dl.Project, prepared_train_dataset_dir='prepared_trainingSample'):
    dataset = project.datasets.create('MNIST_Dataset')
    dataset.to_df()
    _ = dataset.items.upload(local_path=f'{prepared_train_dataset_dir}',
                             local_annotations_path=f'{prepared_train_dataset_dir}/json')
    dataset.add_labels(label_list=[str(i) for i in range(10)])

    # update dataset metadata
    # dataset.metadata['system']['subsets'] = {
    #     'train': json.dumps(dl.Filters(field='dir', values='/train/*').prepare()),
    #     'validation': json.dumps(dl.Filters(field='dir', values='/validation/*').prepare()),
    # }
    # dataset.update(system_metadata=True)
    dataset.open_in_web()


def main():
    project = dl.projects.get(project_id='fb44c2af-9881-446b-a128-3b5e548042c2')

    # split_training_dataset()
    upload_train_dataset_without_json(project=project)
    # upload_train_dataset()


if __name__ == '__main__':
    main()
